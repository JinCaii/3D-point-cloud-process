#直连zivid相机，用于测试
#使用 Zivid SDK 2.11 和 OpenCV 检测 ArUco 标记并估计位姿。
#从3D图像中读取2D图像
#用于continuous Motion
#没有用点云的xyz或深度图来计算相机位姿，而是从3D采集结果里取出2D彩色图


import sys

import cv2
import numpy as np
import zivid
import zivid.experimental.calibration


def main():
    #初始化aruco detector
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    #连接zivid相机
    with zivid.Application() as app:
        print("Connecting to Zivid camera...")
        with app.connect_camera() as camera:
            print(f"Connected: {camera.info}")

            settings = zivid.Settings(    #采集3D图像
                acquisitions=[zivid.Settings.Acquisition()]
            )
            #获取相机内参和畸变系数
            intrinsics = zivid.experimental.calibration.intrinsics(camera, settings)
            matrix = intrinsics.camera_matrix
            distortion = intrinsics.distortion

            #构造内参矩阵
            camera_matrix = np.array(
                [[matrix.fx, 0.0, matrix.cx],
                 [0.0, matrix.fy, matrix.cy],
                 [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )

            #畸变系数5*1向量
            dist_coeffs = np.array(
                [distortion.k1, distortion.k2, distortion.p1,
                 distortion.p2, distortion.k3],
                dtype=np.float64,
            ).reshape(5, 1)

            #创建图像显示窗口
            window_name = "ArUco Detection"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)

            #定义aruco的真实三维坐标
            aruco_length = 0.03    
            half_length = aruco_length / 2.0
            #角点顺序和SOLVEPNP_IPPE_SQUARE的要求一致。
            object_points = np.array(
                [[-half_length, half_length, 0.0],    #左上
                 [half_length, half_length, 0.0],     #右上
                 [half_length, -half_length, 0.0],    #右下
                 [-half_length, -half_length, 0.0]],  #左下
                dtype=np.float32,
            )

            while True:
                #调用zivid进行一次3D采集
                with camera.capture(settings) as captured_frame:
                    bgra = captured_frame.point_cloud().copy_data("bgra")  #从3D点云中读取bgra并复制到numpy数组中。
                    frame = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)  #bgra转换为bgr图像，frame用于后续绘制边框、坐标轴等
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #bgr再转为灰度图
                marker_corners, marker_ids, _ = detector.detectMarkers(gray)
                detected_count = 0 if marker_ids is None else len(marker_ids)

                if detected_count > 0:
                    cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
                    for corners, marker_id in zip(marker_corners, marker_ids.flatten()):
                        pose_found, rvec, tvec = cv2.solvePnP(
                            object_points,
                            corners.reshape(4, 2),
                            camera_matrix,
                            dist_coeffs,
                            flags=cv2.SOLVEPNP_IPPE_SQUARE,
                        )
                        if not pose_found:
                            continue
                        print(f"Marker ID: {marker_id}")
                        print(f"rvec: {rvec.ravel()}")
                        print(f"tvec (m): {tvec.ravel()}")
                        cv2.drawFrameAxes(
                            frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1
                        )

                cv2.putText(
                    frame, f"DICT_4X4_250 | Detected: {detected_count}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
                cv2.putText(
                    frame, "Q / Esc / close window to exit", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(10) & 0xFF
                if (key in (ord("q"), ord("Q"), 27)
                        or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1):
                    break
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except KeyboardInterrupt:
        exit_code = 0
    except Exception as error:
        print(f"Zivid/OpenCV error: {error}", file=sys.stderr)
        print(
            "Check camera power/network and disconnect it in Zivid Studio before retrying.",
            file=sys.stderr,
        )
        exit_code = 1
    finally:
        cv2.destroyAllWindows()
    sys.exit(exit_code)

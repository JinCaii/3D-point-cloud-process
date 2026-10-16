#使用 Zivid SDK 2.11 和 OpenCV 检测 ArUco 标记并估计位姿。

# 依赖：与本机 Zivid SDK 匹配的 zivid Python 包、numpy、opencv-contrib-python。
# 运行：python aruco_3D_detection.py


import sys

import cv2
import numpy as np
import zivid
import zivid.experimental.calibration


def main():
    #连接相机并显示实时检测结果；平移向量的单位为米。
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # Application 必须在连接和采集期间保持存活。
    with zivid.Application() as app:
        print("Connecting to Zivid camera...")
        with app.connect_camera() as camera:
            print(f"Connected: {camera.info}")

            settings = zivid.Settings(
                acquisitions=[zivid.Settings.Acquisition()]
            )
            intrinsics = zivid.experimental.calibration.intrinsics(camera, settings)
            matrix = intrinsics.camera_matrix
            distortion = intrinsics.distortion
            camera_matrix = np.array(
                [[matrix.fx, 0.0, matrix.cx],
                 [0.0, matrix.fy, matrix.cy],
                 [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            dist_coeffs = np.array(
                [distortion.k1, distortion.k2, distortion.p1,
                 distortion.p2, distortion.k3],
                dtype=np.float64,
            ).reshape(5, 1)

            window_name = "ArUco Detection"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)

            aruco_length = 0.03
            half_length = aruco_length / 2.0
            # 以中心为原点，按左上、右上、右下、左下排列。
            # 顺序与检测角点和 SOLVEPNP_IPPE_SQUARE 的要求一致。
            object_points = np.array(
                [[-half_length, half_length, 0.0],
                 [half_length, half_length, 0.0],
                 [half_length, -half_length, 0.0],
                 [-half_length, -half_length, 0.0]],
                dtype=np.float32,
            )

            while True:
                with camera.capture(settings) as captured_frame:
                    bgra = captured_frame.point_cloud().copy_data("bgra")
                    # 生成独立 BGR 图像，不修改 SDK 图像数据。
                    frame = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

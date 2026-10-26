#直连zivid相机，用于测试
#使用 Zivid SDK 2.11 和 OpenCV 检测 ArUco 标记并估计位姿。
#单独用2D图像做位姿估计
#用于stop-and-go

import sys

import cv2
import numpy as np
import zivid  #控制zivid相机
import zivid.experimental.calibration  #读取zivid相机内参


def main():
   #初始化aruco detector
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)  #选择aruco字典，编码0~249
    parameters = cv2.aruco.DetectorParameters()  #使用openCV默认的aruco检测参数
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)  #创建检测器，调用字典和参数

    #连接zivid相机
    with zivid.Application() as app:
        print("Connecting to Zivid camera...")
        with app.connect_camera() as camera:
            print(f"Connected: {camera.info}")

            settings = zivid.Settings2D(  #单独采集2D图像
                acquisitions=[zivid.Settings2D.Acquisition()]
            )

            #获取相机内参和畸变系数
            intrinsics = zivid.experimental.calibration.intrinsics(camera, settings) 
            matrix = intrinsics.camera_matrix #相机内参
            distortion = intrinsics.distortion  #畸变系数

            #构造内参矩阵
            camera_matrix = np.array(
                [[matrix.fx, 0.0, matrix.cx],
                 [0.0, matrix.fy, matrix.cy],
                 [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            #构造畸变系数5*1向量
            dist_coeffs = np.array(
                [distortion.k1, distortion.k2, distortion.p1,
                 distortion.p2, distortion.k3],
                dtype=np.float64,
            ).reshape(5, 1)

            #创建图像显示窗口
            window_name = "ArUco Detection"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600) #窗口大小为800*600，与分辨率无关

            #定义aruco的真实三维坐标
            aruco_length = 0.03   #这里的边长为0.03m，自己修改
            half_length = aruco_length / 2.0
            #角点的顺序和SOLVEPNP_IPPE_SQUARE的要求一致
            object_points = np.array(
                [[-half_length, half_length, 0.0],    #左上
                 [half_length, half_length, 0.0],     #右上
                 [half_length, -half_length, 0.0],    #右下
                 [-half_length, -half_length, 0.0]],  #左下
                dtype=np.float32,
            )

            
            
            while True:  #持续循环，直到满足后面的退出条件
                #调用zivid进行一次2D采集，返回的captured_frame是一个2D采集图像
                with camera.capture(settings) as captured_frame:   
                    bgra = captured_frame.image_bgra().copy_data()   #提取bgra图像并把数据复制到numpy数组中
                    frame = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)   #把bgra图像转换为bgr图像，frame后续可以用于绘制边框、坐标轴等
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)       #再转换为单通道灰度图，grey用于后续检测marker
                
                #从灰度图中读取marker id和角点
                marker_corners, marker_ids, _ = detector.detectMarkers(gray)
                detected_count = 0 if marker_ids is None else len(marker_ids) #计算检测数量

                #PnP位姿转换
                if detected_count > 0:
                    cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)  #绘制marker边框
                    #遍历每个marker
                    for corners, marker_id in zip(marker_corners, marker_ids.flatten()):
                        pose_found, rvec, tvec = cv2.solvePnP(   #PnP输入
                            object_points,   #marker四角的真实三维坐标
                            corners.reshape(4, 2),  #对应的四个二维坐标（相机里的）
                            camera_matrix,  #相机内参
                            dist_coeffs,    #畸变系数
                            flags=cv2.SOLVEPNP_IPPE_SQUARE, #专门适用于正方形marker的PnP方法
                        )
                        if not pose_found:   #若求解失败则跳过这个marker，继续求解下一个
                            continue
                        print(f"Marker ID: {marker_id}")    #打印marker id
                        print(f"rvec: {rvec.ravel()}")      #打印计算出的旋转向量
                        print(f"tvec (m): {tvec.ravel()}")  #打印计算出的平移向量
                        #绘制坐标轴，红色x，绿色y，蓝色z，坐标轴的长度为0.1m
                        cv2.drawFrameAxes(  
                            frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1
                        )

                #图像左上角显示检测到的marker数量
                cv2.putText(
                    frame, f"DICT_4X4_250 | Detected: {detected_count}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
                #显示退出提示：q/esc/关闭窗口来退出
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

#异常处理
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

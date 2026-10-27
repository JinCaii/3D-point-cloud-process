#使用 Zivid SDK 2.11和OpenCV检测 ArUco 标记并估计位姿。
#离线从3D.zdf图像中读取2D图像做位姿估计
#用于continuous motion
#按任意键显示下一张图片，按q或esc退出

#注意：
#要修改的参数：23行aruco字典dictionary，74行aruco边长aruco_length
#captures_stop_and_go文件夹中保存的intrinsics_2d.yml内参文件要与拍照时的匹配

#与筛选关键帧的代码配合

import sys
from pathlib import Path  #用于读取本地图像文件
import yaml #用于读取内参yml文件

import cv2
import numpy as np
import zivid  #控制zivid相机
import zivid.experimental.calibration  #读取zivid相机内参


def main():
    app = zivid.Application()  #初始化zivid sdk

   #初始化aruco detector
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)  #选择aruco字典，编码0~249
    parameters = cv2.aruco.DetectorParameters()  #使用openCV默认的aruco检测参数
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)  #创建检测器，调用字典和参数

    #从文件夹captures_continuousMotion中读取zdf文件以及相机内参intrinsics_3d.yml
    #见open3D_tutorial 200Caputre_interval.py的第13行
    # folder = Path(r"C:\Users\Jin Cai\Desktop\captures_continuousMotion")
    # image_files = sorted(folder.glob("*.zdf")) #查找所有的zdf文件并排序


    #若从关键帧文件夹keyframe里读取
    #原始采集文件夹
    folder = Path( r"C:\Users\User\Desktop\captures_continuousMotion")

    #关键帧文件夹
    keyframe_folder = folder / "keyframe"

    #只读取筛选出的关键帧ZDF文件
    image_files = sorted(keyframe_folder.glob("*.zdf"))

    if not image_files:  #若没找到
        raise FileNotFoundError("No zdf files found, please check the path!")


    # 读取之前保存的相机内参
    # intrinsics = zivid.CameraIntrinsics.load(
    #     str(folder / "intrinsics_3d.yml")   #绝对路径
    # )

    #读取之前保存的相机内参
    with open(folder / "intrinsics_3d.yml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    #提取相机内参
    camera_data = data["CameraIntrinsics"]
    matrix = camera_data["CameraMatrix"]
    dist = camera_data["Distortion"]
    
    fx = matrix["FX"]
    fy = matrix["FY"]
    cx = matrix["CX"]
    cy = matrix["CY"]

    # 转换为 OpenCV 相机内参矩阵
    camera_matrix = np.array(
        [[fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]],
        dtype=np.float64
    )
    
    # 转换为 OpenCV 畸变系数
    dist_coeffs = np.array(
        [dist["K1"],
        dist["K2"],
        dist["P1"],
        dist["P2"],
        dist["K3"]],
    dtype=np.float64
        )
    
    print("相机内参矩阵：")
    print(camera_matrix)
    
    print("畸变系数：")
    print(dist_coeffs)        
            
    #创建图像显示窗口
    window_name = "ArUco Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600) #窗口大小为800*600，与分辨率无关

    #定义aruco的真实三维坐标
    aruco_length = 0.1   #这里的边长为0.1m，自己修改
    half_length = aruco_length / 2.0
    #角点的顺序和SOLVEPNP_IPPE_SQUARE的要求一致
    object_points = np.array(
        [[-half_length, half_length, 0.0],    #左上
         [half_length, half_length, 0.0],     #右上
         [half_length, -half_length, 0.0],    #右下
         [-half_length, -half_length, 0.0]],  #左下
        dtype=np.float32,
            )

    #遍历文件夹中所有的zdf图像，全部读取完后自动结束
    for image_path in image_files:
        print(f"\nin processing: {image_path.name}")

        #读取本地zdf图像
        try:
            zivid_frame = zivid.Frame(str(image_path))
        
            #从3D点云中提取2D的rgba
            point_cloud = zivid_frame.point_cloud()
            rgba = point_cloud.copy_data("rgba")
        
            #将rgba转换为openCV使用的bgr格式
            frame = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        
        #若当前图像读取失败，则跳过并读取下一个
        except Exception as error:
            print(f"Cannot read .zdf image: {image_path.name}")
            print(error)
            continue
    
        #转换为灰度图，用于marker检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
        #从灰度图中读取marker id和角点
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)
        detected_count = 0 if marker_ids is None else len(marker_ids) #计算检测数量

        #PnP位姿转换
        if detected_count > 0:
            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)  #绘制marker边框和ID
            #遍历当前图像中的每个marker
            for corners, marker_id in zip(marker_corners, marker_ids.flatten()):
                pose_found, rvec, tvec = cv2.solvePnP(   #PnP输入
                    object_points,              #marker四角的真实三维坐标
                    corners.reshape(4, 2),      #对应的四个二维坐标（相机里的）
                    camera_matrix,              #相机内参
                    dist_coeffs,                #畸变系数
                    flags=cv2.SOLVEPNP_IPPE_SQUARE, #专门适用于正方形marker的PnP方法
                )
                
                if not pose_found:   #若求解失败则跳过这个marker，继续求解下一个
                    continue

                print(f"Marker ID: {marker_id}")    #打印marker id
                print(f"rvec: {rvec.ravel()}")      #打印计算出的旋转向量
                print(f"tvec (m): {tvec.ravel()}")  #打印计算出的平移向量


                #将旋转向量转换为3×3旋转矩阵
                R, _ = cv2.Rodrigues(rvec)
                T_marker_to_camera = np.eye(4, dtype=np.float64) #构建4x4齐次变换阵
                T_marker_to_camera[:3, :3] = R  #左上角3×3填入旋转矩阵
                T_marker_to_camera[:3, 3] = tvec.flatten() #最后一列填入平移向量

                print(T_marker_to_camera)

                #创建保存矩阵的文件夹
                matrix_folder = folder / "transformation_matrices"
                matrix_folder.mkdir(exist_ok=True)
                matrix_path = matrix_folder / f"{image_path.stem}_marker_{marker_id}.npy"   #设置保存的文件名
                np.save(matrix_path, T_marker_to_camera)  #保存变换阵
                print(f"Matrix saved to: {matrix_path}")

                #这里的变换阵是marker -> camera，后续可能需要camera -> marker，则求逆
                #T_camera_to_marker = np.linalg.inv(T_marker_to_camera)

                #绘制坐标轴，红色x，绿色y，蓝色z，坐标轴的长度为0.1m
                cv2.drawFrameAxes(  
                    frame, 
                    camera_matrix, 
                    dist_coeffs, 
                    rvec, 
                    tvec, 
                    0.1
                )

        #图像左上角显示检测到的marker数量
        cv2.putText(
            frame, f"DICT_4X4_250 | Detected: {detected_count}",
            (10, 25), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1,   #字体大小
            (0, 255, 0), 
            2,    #字体粗细
        )

        #显示当前zdf图像的文件名
        cv2.putText(
            frame,
            image_path.name,
            (10,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            1
        )

        #显示操作提示
        cv2.putText(
            frame,
            "Any key: next | Q / Esc: exit",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            1
        )
        
        #显示检测结果
        cv2.imshow(window_name, frame)
        
        #一直等待按键，直到按了任意键才切换下一张图片
        key = cv2.waitKey(0) & 0xFF  
        
        if (
            key in (ord("q"), ord("Q"), 27)
            or cv2.getWindowProperty(
                window_name, cv2.WND_PROP_VISIBLE
            ) < 1
        ):
            break
        
    print("\nImage processing completed!")
        
    

# 异常处理
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f"Zivid/OpenCV error: {error}", file=sys.stderr)
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()

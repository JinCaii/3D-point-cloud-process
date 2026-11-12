#使用 Zivid SDK 2.11 和OpenCV检测ArUco标记并估计位姿。
#单独用2D图像做位姿估计
#用于stop-and-go
#按任意键显示下一张图片，按q或esc退出
#use Zivid SDK 2.11 and OpenCV to do marder detection and pose estimation
#use only 2D images to do pose estimation
#for stop-and-go
#press any key to display the next image, or press Q or Esc to exit

#注意：
#要修改的参数：22行aruco字典dictionary，74行aruco边长aruco_length
#captures_stop_and_go文件夹中保存的intrinsics_2d.yml内参文件要与拍照时的匹配
#note:
#parameters to modify: ArUco dictionary (line 22) and ArUco marker size (line 74)
#the intrinsics_2d.yml file saved in the captures_stop_and_go folder must match the camera settings used during image acquisition

import sys
from pathlib import Path  
import yaml  #读取内参的yml文件read camera intrincis from a YAML file

import cv2
import numpy as np
import zivid  #控制zivid相机
import zivid.experimental.calibration  #读取zivid相机内参


def main():
   #初始化aruco detector
   #initialize the aruco detector
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)  #选择aruco字典，编码0~249 #select the aruco dictionary
    parameters = cv2.aruco.DetectorParameters()  #使用openCV默认的aruco检测参数  #use the default openCV ArUco detection parameters
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)  #创建检测器，调用字典和参数  #create the detector using the specified dictionary and parameters
    #从文件夹captures_stop_and_go中读取图像以及相机内参intrinsics_2d.yml
    #read images and camera intrinsics (intrinsics_2d.yml) from the captures_stop_and_go folder.
    #见open3D_tutorial 100Caputre_2D_3D.py的第12行
    folder = Path(r"D:\captures_stop_and_go\forward")
    image_files = sorted(folder.glob("capture_*.png")) #查找所有.png图像，并按文件名排序 #find all png images and sort them by filename

    if not image_files:  #若没找到 #if no images are found
        raise FileNotFoundError("No png files found, please check the path!")

    #读取之前保存的相机内参  #load the previously saved camera intrinsics
    with open(folder.parent / "intrinsics_2d.yml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    #提取相机内参 #extract the camera intrinsics
    camera_data = data["CameraIntrinsics"]
    matrix = camera_data["CameraMatrix"]
    dist = camera_data["Distortion"]

    fx = matrix["FX"]
    fy = matrix["FY"]
    cx = matrix["CX"]
    cy = matrix["CY"]

    #转换为openCV相机内参矩阵 #convert to the openCV camera intrinsic matrix
    camera_matrix = np.array(
        [[fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]],
        dtype=np.float64
    )

    #转换为openCV畸变系数  #convert to openCV distortion coefficients
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
            
    #创建图像显示窗口  #create a window to display images
    window_name = "ArUco Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600) #窗口大小为800*600，与分辨率无关

    #定义aruco的真实三维坐标  #the actual 3D coordinates of the ArUco marker
    aruco_length = 0.1   #这里的边长为0.1m  #the marker side length is 0.1m
    half_length = aruco_length / 2.0
    #角点的顺序和SOLVEPNP_IPPE_SQUARE的要求一致 
    #the corner order follows the requirements of SOLVEPNP_IPPE_SQUARE
    object_points = np.array(
        [[-half_length, half_length, 0.0],    #左上 top-left
         [half_length, half_length, 0.0],     #右上 top-right
         [half_length, -half_length, 0.0],    #右下 bottom-right
         [-half_length, -half_length, 0.0]],  #左下 bottom-left
        dtype=np.float32,
            )

    #遍历文件夹中所有的PNG图像，全部读取完后自动结束
    #iterate through all png images in the folder and automatically exit after processing all images
    for image_path in image_files:
        print(f"\nin processing: {image_path.name}")

        #读取本地PNG图像
        #read the local png images
        frame = cv2.imread(str(image_path))

        #若无法读取图像
        #skip the image is it cannot be loaded
        if frame is None:
            print(f"cannot read the iamge: {image_path.name}")
            continue
    
        #转换为灰度图，用于marker检测
        #convert the image to grayscale for marker detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
        #从灰度图中读取marker id和角点
        #read marker id and corners in the grayscale image
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)
        detected_count = 0 if marker_ids is None else len(marker_ids) #计算检测数量  #count the number of detected markers

        #只用id最小的marker做位姿估计
        #if there are two or more markers in one image
        #only use the marker with smallest id to estimate pose
        if marker_ids is not None:
            min_idx = np.argmin(marker_ids.flatten())   #找到最小ID对应的索引       #find the index of the marker with the smallest ID
            marker_corners = [marker_corners[min_idx]]  #只保留该marker的四个角点   #keep only the four corners of this marker
            marker_ids = marker_ids[min_idx:min_idx+1]  #只保留该marker的ID         #keep only the id of this marker
        detected_count = 0 if marker_ids is None else len(marker_ids)

        #PnP位姿转换 
        #PnP pose estimation
        if detected_count > 0:
            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)  #绘制marker边框和ID  #draw the boundaries and ids of the detected marker
            #遍历当前图像中的每个marker
            #Iterate through each marker in the current image
            for corners, marker_id in zip(marker_corners, marker_ids.flatten()):
                pose_found, rvec, tvec = cv2.solvePnP(   #PnP输入  #PnP inputs
                    object_points,              #marker四角的真实三维坐标       #real 3D coordinates of the four marker corners
                    corners.reshape(4, 2),      #对应的四个二维坐标（相机里的）  #Corresponding 2D corner coordinates in the image
                    camera_matrix,              #相机内参                      #camera intrinsic matrix
                    dist_coeffs,                #畸变系数                      #distortin coefficients
                    flags=cv2.SOLVEPNP_IPPE_SQUARE, #专门适用于正方形marker的PnP方法    #PnP method specifically designed for square markers
                )
                
                if not pose_found:   #若求解失败则跳过这个marker，继续求解下一个
                    continue         #if pose estimation fails, skip this marker and continue with the next one

                print(f"Marker ID: {marker_id}")    #打印marker id          #print the marker ID.
                print(f"rvec: {rvec.ravel()}")      #打印计算出的旋转向量     #print the estimated rotation vector
                print(f"tvec (m): {tvec.ravel()}")  #打印计算出的平移向量     #print the estimated translation vector

                #将旋转向量转换为3×3旋转矩阵
                #convert the rotation vector into a 3×3 rotation matrix
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
                #this transformation matrix represents marker -> camera. If camera -> marker is needed later, compute its inverse.

                #绘制坐标轴，红色x，绿色y，蓝色z，坐标轴的长度为0.1m
                #draw the coordinate axes: red for X, green for Y, and blue for Z, with an axis length of 0.1 m.
                cv2.drawFrameAxes(  
                    frame, 
                    camera_matrix, 
                    dist_coeffs, 
                    rvec, 
                    tvec, 
                    0.1
                )

        #图像左上角显示检测到的marker数量
        #display the number of detected markers in the upper-left corner of the image
        cv2.putText(
            frame, f"DICT_4X4_250 | Detected: {detected_count}",
            (10, 25), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1,    #字体大小
            (0, 255, 0), 
            2,      #字体粗细
        )

        #显示当前图像文件名
        #display the current image filename
        cv2.putText(
            frame,
            image_path.name,
            (10,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        #显示操作提示
        #display keyboard instructions
        cv2.putText(
            frame,
            "Any key: next | Q / Esc: exit",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        #显示检测结果
        #display the detectiion results
        cv2.imshow(window_name, frame)
        
        #一直等待按键，直到按了任意键才切换下一张图片
        #wait for a keypress before displaying the next image
        key = cv2.waitKey(0) & 0xFF  
        
        if (
            key in (ord("q"), ord("Q"), 27)
            or cv2.getWindowProperty(
                window_name, cv2.WND_PROP_VISIBLE
            ) < 1
        ):
            break
        
    print("\nImage processing completed!")
        
    

#异常处理
#exception handling
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

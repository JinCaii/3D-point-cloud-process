#对应的是stop-and-go的位姿估计代码
#使用openCV验证估计用marker detection和pnp算法估计位姿的准确性
#读取已保存的png图像和相机内参
#使用pnp估计位姿，并将marker的4个3D角点重新投影到2D图像
#对比重新投影的角点与png图像里的原角点，计算重投影误差和RMSE
#按任意键显示下一张图像，按q或esc退出

#验证结果保存在reprojection_validation文件夹中，包含每张图像的验证结果图像和所有marker的误差csv文件
#rmse小于0.2拟合良好，0.2~0.5拟合一般，0.5以上拟合较差

import sys
import csv
from pathlib import Path

import cv2
import numpy as np
import yaml


def main():
    #初始化aruco detector
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
   
    #读取png图像和相机内参
    folder = Path(r"D:\captures_stop_and_go")   #修改为自己的文件夹路径
    image_files = sorted(folder.glob("capture_*.png"))

    if not image_files:
        raise FileNotFoundError("No PNG files found. Please check the folder path!")

    #读取相机内参
    with open(
        folder / "intrinsics_2d.yml",
        "r",
        encoding="utf-8"
    ) as f:
        data = yaml.safe_load(f)

    camera_data = data["CameraIntrinsics"]

    matrix = camera_data["CameraMatrix"]
    dist = camera_data["Distortion"]

    fx = matrix["FX"]
    fy = matrix["FY"]
    cx = matrix["CX"]
    cy = matrix["CY"]

    camera_matrix = np.array(    #内参矩阵
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ],
        dtype=np.float64
    )

    dist_coeffs = np.array(     #畸变系数
        [
            dist["K1"],
            dist["K2"],
            dist["P1"],
            dist["P2"],
            dist["K3"]
        ],
        dtype=np.float64
    )

    print("Camera intrinsic matrix:")
    print(camera_matrix)

    print("\nDistortion coefficients:")
    print(dist_coeffs)
    
    #marker的真实三维角点
    #注意：必须与原程序保持一致
    aruco_length = 0.1  #100mm，单位为m

    half_length = aruco_length / 2.0

    #角点的顺序和SOLVEPNP_IPPE_SQUARE的要求一致
    object_points = np.array(
        [
            [-half_length, half_length, 0.0],   # 左上
            [half_length, half_length, 0.0],    # 右上
            [half_length, -half_length, 0.0],   # 右下
            [-half_length, -half_length, 0.0]   # 左下
        ],
        dtype=np.float32
    )

    
    #在原文件夹中创建reprojection_validation文件夹，保存验证结果
    output_folder = folder / "reprojection_validation"

    output_folder.mkdir(parents=True, exist_ok=True)
    #保存所有marker的误差结果为csv文件
    csv_results = []

    #创建验证窗口显示结果
    window_name = "Reprojection Error Validation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        window_name,
        1000,
        700
    )



    #遍历所有png图像，这里和原代码一样
    for image_path in image_files:

        print(f"\nProcessing: {image_path.name}")

        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"Cannot read image: {image_path.name}")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #从灰度图中读取marker id和角点
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)

        detected_count = ( 0 if marker_ids is None else len(marker_ids))  #计算检测数量

        
        #对每个marker进行pnp位姿估计
        if detected_count > 0:

            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
            #遍历当前图像中的每个marker
            for corners, marker_id in zip(marker_corners, marker_ids.flatten()):
                #转换为4*2数组
                detected_points = corners.reshape(4, 2)
                pose_found, rvec, tvec = cv2.solvePnP(
                    object_points,
                    detected_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if not pose_found:
                    print(f"Marker {marker_id}: PnP failed")
                    continue

                #将3D角点重新投影到2D图像
                projected_points, _ = cv2.projectPoints(
                    object_points,
                    rvec,
                    tvec,
                    camera_matrix,
                    dist_coeffs
                )

                projected_points = projected_points.reshape(4, 2)

                #计算每个角点的重投影误差
                #每个角点的二维坐标差
                differences = (detected_points - projected_points)
                #每个角点的欧氏距离，单位为像素pixel
                #L2范数，也就是欧式距离
                #axis=1表示按行计算范数，分别计算四个角点的误差，而不是把所有角点一起计算
                corner_errors = np.linalg.norm(differences,axis=1)  

                #计算整体的RMS重投影误差，及上面四个角点的均方误差
                rmse = np.sqrt(np.mean(corner_errors ** 2))
                #最大角点误差
                max_error = np.max(corner_errors)
                #平均角点误差
                mean_error = np.mean(corner_errors)

            
                #输出结果
                print(f"\nMarker ID: {marker_id}")

                print("Detected corner points (pixels):")    #真实值
                print(detected_points)

                print("Reprojected corner points (pixels):") #重投影值
                print(projected_points)

                print("Corner reprojection errors (pixels):")
                print(corner_errors)

                print(f"Mean error: {mean_error:.4f} px")     #平均误差
                print(f"RMSE:       {rmse:.4f} px")           #rmse
                print(f"Max error:  {max_error:.4f} px")      #最大误差

                #将结果保存到csv_results列表中
                csv_results.append(
                    [
                        #以marker自身的标准朝向为参考，corner0，1，2，3的顺序是左上，右上，右下，左下
                        image_path.name,            #当前png图像的文件名
                        int(marker_id),             #marker id
                        float(corner_errors[0]),    #第一个角点的重投影误差
                        float(corner_errors[1]),    #第二个角点
                        float(corner_errors[2]),    #第三个角点
                        float(corner_errors[3]),    #第四个角点
                        float(mean_error),          #平均误差
                        float(rmse),
                        float(max_error)
                    ]
                )

                #可视化重投影的角点和实际角点
                for i in range(4):
                    # 实际检测角点
                    detected = tuple(
                        np.round(
                            detected_points[i]
                        ).astype(int)
                    )

                    # 重投影角点
                    projected = tuple(
                        np.round(
                            projected_points[i]
                        ).astype(int)
                    )

                    #绿色圆点：实际检测角点
                    cv2.circle(
                        frame,
                        detected,
                        5,
                        (0, 255, 0),
                        -1
                    )

                    #红色十字：重投影角点
                    cv2.drawMarker(
                        frame,
                        projected,
                        (0, 0, 255),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=14,
                        thickness=2
                    )

                    #黄色连线：重投影误差
                    cv2.line(
                        frame,
                        detected,
                        projected,
                        (0, 255, 255),
                        1
                    )

                
                #在marker中心位置显示marker id和rmse误差
                center = np.mean(detected_points, axis=0).astype(int)
                text_position = (int(center[0]), int(center[1]))
                cv2.putText(
                    frame,
                    f"ID {marker_id}: {rmse:.3f} px",
                    text_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    2
                )

        
        #显示图像信息
        cv2.putText(
            frame,
            f"Detected Markers: {detected_count}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            image_path.name,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

        cv2.putText(
            frame,
            "Green: detected | Red: projected",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

        cv2.putText(
            frame,
            "Any key: next | Q / Esc: exit",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

       
        #保存验证的图像
        output_image_path = (
            output_folder /
            f"{image_path.stem}_reprojection.png"
        )
        cv2.imwrite(str(output_image_path), frame)
        print(f"Validation image saved: {output_image_path}")

        #显示当前验证图像
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(0) & 0xFF

        if (
            key in (ord("q"), ord("Q"), 27)
            or cv2.getWindowProperty(
                window_name,
                cv2.WND_PROP_VISIBLE
            ) < 1
        ):
            break

    
    #保存csv误差结果
    csv_path = (output_folder / "reprojection_errors.csv")
    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)
        writer.writerow(
            [
                "Image",
                "Marker_ID",
                "Corner_0_Error_px",
                "Corner_1_Error_px",
                "Corner_2_Error_px",
                "Corner_3_Error_px",
                "Mean_Error_px",
                "RMSE_px",
                "Max_Error_px"
            ]
        )

        writer.writerows(csv_results)

    print("\nReprojection validation completed!")
    print(f"CSV saved to: {csv_path}")

   
    #输出统计结果
    if csv_results:

        all_rmse = np.array(
            [row[7] for row in csv_results]
        )

        print(f"Validated markers: {len(all_rmse)}")
        print(f"Average RMSE: {np.mean(all_rmse):.4f} px")
        print(f"Maximum RMSE: {np.max(all_rmse):.4f} px")
        print(f"Minimum RMSE: {np.min(all_rmse):.4f} px")


#异常处理
if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        pass

    except Exception as error:
        print(
            f"Validation error: {error}",
            file=sys.stderr
        )
        sys.exit(1)

    finally:
        cv2.destroyAllWindows()
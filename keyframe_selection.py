from pathlib import Path
import csv     #生成csv表格
import shutil  #用于复制zdf文件
import cv2
import numpy as np
import zivid



#读取zdf图像的路径，即之前相机拍摄保存下来的
folder = Path(
    r"C:\Users\User\Desktop\captures_continuousMotion"
)

#创建keyframe文件夹，用于保存筛选结果
keyframe_folder = folder / "keyframe"   #在原文件夹下建一个子文件夹
keyframe_folder.mkdir(parents=True, exist_ok=True)   #若该文件夹已存在也ok

#csv文件保存路径
output_csv = keyframe_folder / "best_marker_clouds.csv"

#---------------------------------------------------------------------------------------

#初始化aruco检测器
dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_250     #aruco字典
)
parameters = cv2.aruco.DetectorParameters()   #使用openCV默认的marker检测参数
detector = cv2.aruco.ArucoDetector(     #创建aruco检测器，后续调用detector.detectMarkers()即可检测 Marker
    dictionary, parameters
)

#---------------------------------------------------------------------------------------

#选择keyframe
#每个Marker ID可能有多张图像，从中选择最佳的一张
best_markers = {}

#读取所有ZDF文件并排序
zdf_files = sorted(folder.glob("capture_*.zdf"))

if not zdf_files:   #若没找到
    raise FileNotFoundError("No zdf files found, please check the path!")


#创建zivid application遍历每个zdf文件
with zivid.Application():   #不是用来连接相机的，而是为离线读取zdf提供sdk环境

    for zdf_path in zdf_files:  #遍历存储zdf的文件夹直到全部读取完

        #从3D点云中提取rgba
        with zivid.Frame(zdf_path) as frame:
            image = frame.point_cloud().copy_data("rgba")

        #将rgba转换为openCV使用的bgr
        image = cv2.cvtColor(
            image, cv2.COLOR_RGBA2BGR
        )

        #--------------------------------
        #真实情况不一定在整个图像的中心，而是width在中心就行！！！！
        #-------------------------------
        
        #获取图像的尺寸
        height, width = image.shape[:2]

        #计算图像中心，作为reference
        image_center = np.array([
            width / 2,
            height / 2
        ])

        #读取marker的id的角点
        corners, ids, rejected = detector.detectMarkers(image)

        #若没读取到
        if ids is None:  
            print(f"{zdf_path.name}: no marker detected")
            continue

        #遍历当前图像中所有检测到的marker
        for marker_corners, marker_id in zip(
            corners, ids.flatten()
        ):

            marker_id = int(marker_id)

            #获取marker四个角点的像素坐标
            points = marker_corners.reshape(4, 2)

            #计算marker中心  也可能是只计算width
            marker_center = np.mean(points, axis=0)

            #计算marker中心到图像中心的距离，这里是欧式距离
            distance = np.linalg.norm(
                marker_center - image_center
            )

            #如果当前marker id没有记录，或者当前帧的中心距离更小，则更新记录
            if (
                marker_id not in best_markers    
                or distance < best_markers[marker_id]["distance"]
            ):

                #保存当前最佳的文件名和距离
                best_markers[marker_id] = {
                    "zdf": zdf_path.name,
                    "distance": float(distance)
                }

        print(
            f"{zdf_path.name}: 检测到 {len(ids)} 个Marker"
        )


#-------------------------------------------------------------------

#输出最佳关键帧的信息
print("\n最佳点云筛选结果")

for marker_id in sorted(best_markers):

    result = best_markers[marker_id]

    print(
        f"Marker ID: {marker_id}, "
        f"ZDF: {result['zdf']}, "
        f"中心距离: {result['distance']:.2f} px"
    )

#获取所有最佳zdf文件名，并去除重复项
#去除重复项：因为同一张zdf可能同时包含多个marker，而且可能恰好是多个marker各自的最佳帧  
#可能性很小
selected_zdfs = {
    result["zdf"] for result in best_markers.values()
}

#将最佳zdf复制到keyframe文件夹
for zdf_name in sorted(selected_zdfs):

    source = folder / zdf_name
    destination = keyframe_folder / zdf_name

    shutil.copy2(source, destination)

    print(f"已复制：{zdf_name}")


#保存csv文件
with open(
    output_csv, "w",  #写入
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.writer(file)

    #表头
    writer.writerow([
        "Marker ID",
        "Best ZDF",
        "Center Distance (px)"
    ])

    #每个marker对应一行
    for marker_id in sorted(best_markers):

        result = best_markers[marker_id]

        writer.writerow([
            marker_id,
            result["zdf"],
            round(result["distance"], 2)
        ])

print("\n筛选完成！")
print(f"检测到的Marker数量：{len(best_markers)}")
print(f"复制的ZDF文件数量：{len(selected_zdfs)}")
print(f"结果保存位置：{keyframe_folder}")

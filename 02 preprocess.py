#用forward里的所有文件夹

import open3d as o3d
import numpy as np
from pathlib import Path

#原始点云所在文件夹
ply_path = Path(r"D:\captures_stop_and_go\forward")

#深度过滤参数，单位 mm，用于背景去除
z_min = 1100
z_max = 1500

#体素下采样参数，单位mm
voxel_size = 0.5

#离群点去除参数
nb_neighbors = 20   #计算每个点的平均临近距离时，考虑最近的20个点
std_ratio = 2.0     #设2倍标准差来确定离群点的判定阈值。这里的数值越低，滤波越严格，得到的红色离群点也就越多

#创建一个空字典，用于保存处理后的点云对象
processed_pcds = {}

#创建深度过滤、下采样和SOR的输出文件夹，在原文件夹forward里
filtered_folder = ply_path / "filtered"
downsampled_folder = ply_path / "downsampled"
sor_folder = ply_path / "SOR"

#如果文件夹不存在，则自动创建
filtered_folder.mkdir(parents=True, exist_ok=True)
downsampled_folder.mkdir(parents=True, exist_ok=True)
sor_folder.mkdir(parents=True, exist_ok=True)

#依次处理forward文件夹里的所有原始PLY文件
ply_files = sorted(ply_path.glob("capture_*.ply"))

n = 0  #计数器

for file_path in ply_files:

    file_name = file_path.name  #提取path中的完整文件名，包含扩展名
    file_stem = file_path.stem  #不包含扩展名，例如capture_003

    print(f"\n----------Processing {file_name} ----------")

    #读取原始点云
    pcd = o3d.io.read_point_cloud(str(file_path))
    print("original point count:", len(pcd.points))

    #显示原始点云
    o3d.visualization.draw_geometries([pcd], window_name=f"{file_stem}_rawdata")

   #深度过滤，去除背景------------------------------------------------------------------------------------------------------------
    xyz = np.asarray(pcd.points)    #读取xyz坐标信息，并转换为numpy数组，每一行代表一个点的xyz坐标
    rgb = np.asarray(pcd.colors)    #读取rgb颜色信息，并转换为numpy数组，每一行对应一个点的颜色信息

    points = xyz    #xyz信息保存到变量points中
    colors = rgb    #rgb信息保存到变量colors中

    #mask是numpy中的布尔掩码，可以理解为一个筛选器，满足条件的标为true，不满足的标为false
    #用于判断哪些点位于设置的z轴深度范围内，只保留符合要求的点
    mask = (np.isfinite(points).all(axis=1)& (points[:, 2] >= z_min)& (points[:, 2] <= z_max))   #只用比较第3列，即z轴的坐标

    #numpy的布尔索引操作，只保留mask筛选后标记为true的点
    filtered_points = points[mask]
    filtered_colors = colors[mask]

    #创建过滤后的点云
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    print("point cloud after depth filtering:", len(filtered_pcd.points))

    #显示过滤后的点云
    o3d.visualization.draw_geometries([filtered_pcd], window_name=f"{file_stem}_backgroundRemoved")

    #保存深度过滤后的点云
    filtered_path = filtered_folder / f"{file_stem}_filtered.ply"
    filtered_saved = o3d.io.write_point_cloud(str(filtered_path), filtered_pcd)

    #体素下采样--------------------------------------------------------------------------------------------------------------------------
    #读取背景过滤后的点云进行下采样
    downpcd = filtered_pcd.voxel_down_sample(voxel_size=voxel_size)
    print("point cloud after voxel downsampling:", len(downpcd.points))

    #显示下采样后的点云
    o3d.visualization.draw_geometries([downpcd], window_name=f"{file_stem}_downsampled")

    #保存下采样后的点云
    downsampled_path = downsampled_folder / f"{file_stem}_downsampled.ply"
    downsampled_saved = o3d.io.write_point_cloud(str(downsampled_path), downpcd)

    #SOR统计离群点去除------------------------------------------------------------------------------------------------------------------
    #用体素下采样后的点进行离群点去除
    #cl：去除离群点后保留下来的点云，即inlier
    #ind：这些保留下来的点在原点云中的index
    #SOR可能会把pin的边缘或尖端等局部密度较低的有效点误判为离群点
    cl, ind = downpcd.remove_statistical_outlier(nb_neighbors, std_ratio)
    print("Point count after SOR:", len(cl.points))
    print("Number of removed outliers:", len(downpcd.points) - len(cl.points))

    #可视化SOR处理后的点云
    inlier_cloud = downpcd.select_by_index(ind)
    outlier_cloud = downpcd.select_by_index(ind, invert=True)

    inlier_cloud.paint_uniform_color([0, 1, 0])     #保留的点是绿色
    outlier_cloud.paint_uniform_color([1, 0, 0])    #去除的离群点是红色

    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name=f"{file_stem}_SOR_inliers_outliers")
    o3d.visualization.draw_geometries([cl], window_name=f"{file_stem}_SOR")

    #保存SOR处理后的点云
    sor_path = sor_folder / f"{file_stem}_sor.ply"
    sor_saved = o3d.io.write_point_cloud(str(sor_path), cl)

    #保存成功后计数器+1（所有阶段都成功才计数）
    if filtered_saved and downsampled_saved and sor_saved:
        n += 1
        print(f"{file_name} processed and saved successfully!")
    else:
        print(f"Failed to save {file_name}!")

    #保存到字典，方便后续配准使用
    processed_pcds[file_stem] = downpcd

    # 后续点云配准时可以直接写：
    # source = processed_pcds["capture_003"]  source和target指向已完成预处理的点云
    # target = processed_pcds["capture_004"]
    # o3d.visualization.draw_geometries([source, target])

print(f"\nProcessing complete! Successfully saved {n} point clouds!")
#在平修正后的结果上，进一步利用点云之间的几何对应关系进行精配准
#点到面ICP
#步骤：估计点云法向量 -> 按顺序对不同的capture做ICP
#用的是顺序配准，capture004和003，003和002，002和001配准，而不是004直接配准到001

import copy
import numpy as np
import open3d as o3d
from pathlib import Path


folder = Path(r"D:\captures_stop_and_go\forward")
#用平面修正后的结果
plane_folder = folder / "plane_based_correction"

output_folder = folder / "ICP_Point_to_Plane_after_plane_correction"
output_matrix_folder = output_folder / "transformation_matrices"
output_folder.mkdir(parents=True, exist_ok=True)
output_matrix_folder.mkdir(parents=True, exist_ok=True)


#ICP参数设置
max_correspondence_distance = 5     #最大对应点阈值，是两个点之间的欧氏距离
max_iterations = 4                  #迭代次数

#估计法向量的参数，mm
#normal estimation parameters, mm
normal_radius = 10.0    #搜索半径为10mm内的邻域点
normal_max_nn = 30      #估计法向量时，最多使用30个邻域点


#ICP配准的评估函数-------------------------------------------------------------------------------------------
#evaluate for the ICP registration---------------------------------------------------------------------------------
#输出：fitness和RMSE
def evaluate(source, target, transform):
    return o3d.pipelines.registration.evaluate_registration(
        source,
        target,
        max_correspondence_distance,
        transform
    )


#可视化
#visualization----------------------------------------------------------------
def show_alignment(point_clouds, title, use_capture_colors=False):
    displayed = [copy.deepcopy(pcd) for pcd in point_clouds]

    if use_capture_colors:
        colors = [
            [1, 0, 0],       #capture001:red
            [0, 1, 0],       #capture001:green
            [0, 0, 1],       #capture001:blue
            [1, 0.706, 0]    #capture001:orange
        ]

        for pcd, color in zip(displayed, colors):
            pcd.paint_uniform_color(color)

    o3d.visualization.draw_geometries(
        displayed,
        window_name=title
    )


#读取基于平面修正的结果
#load plane-based correction point clouds and transformations
pcds = []               #保存修正过的点云
plane_global = []       

for i in range(1, 5):
    ply_path = (plane_folder /f"capture_{i:03d}_plane_corrected.ply")
    matrix_path = (plane_folder /f"capture_{i:03d}_final.npy")  #用的修正后的final matrix
    if not ply_path.is_file():  #若点云文件不存在
        raise FileNotFoundError(ply_path)
    if not matrix_path.is_file(): #若matrix文件不存在
        raise FileNotFoundError(matrix_path)

    #读取ply点云文件
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        raise ValueError(f"Empty point cloud: {ply_path}")

    #读取修正后的final matrix
    #load complete plane-based Correction transformation
    transform = np.load(matrix_path)
    #检查矩阵是否合法
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"Invalid transformation: {matrix_path}")
    if not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError(f"Invalid homogeneous matrix: {matrix_path}")

    #将点云和变换矩阵分别添加到对应的列表
    pcds.append(pcd)
    plane_global.append(transform)

    print(f"Loaded capture {i:03d}: {len(pcd.points)} points")

#估计法向量，点到面ICP需要目标平面的法向量
#estimate normal vectors -------------------------------------------------------------
#point-to-plane ICP needs the normal vectors of target point clouds
print("\n-------Estimating normals-------")
for i, pcd in enumerate(pcds, start=1): #从序号1开始遍历，也就是capture002
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(      #混合邻域搜索方法，KDT-Tree是一种数据结构
            radius=normal_radius,   #同时限制搜索半径，一开始自定义的10mm
            max_nn=normal_max_nn    #以及邻域点的数量，一开始自定义的30个邻域点
        )
    )

    pcd.normalize_normals() #归一化为单位向量
    print(f"Normals estimated for capture {i:03d}")


#sequential point-to-plane ICP------------------------------------------------
#global reference = capture001
#全局参考是cpature001
#all point clouds are already in the common coordinate frame
#由于之前的配准，所有的点云已经在共同坐标系下了
#pairwise registration:
#002 -> 001
#003 -> 002
#004 -> 003

refined_global = [np.eye(4, dtype=np.float64)]  #创建一个4*4的单位阵，是capture001相对于自己的变换

for i in range(1, len(pcds)):
    source = pcds[i]        #cpature002,003,004
    target = pcds[i - 1]    #capture001,002,003，target是前一个点云

    #创建4*4单位阵，作为ICP的初始变换
    #因为四组点云经过之前的初配准以及修正，理论上处于同一个坐标系。单位阵即为初始猜测
    #因此ICP只需要估计剩余的幅度误差，小幅度修正单位阵
    initial_pair = np.eye(4, dtype=np.float64)

    #评估ICP配准前的点云
    #evaluate before ICP, fitness and RMSE
    before = evaluate(source, target, initial_pair)

    #Point-to-Plane ICP
    result = o3d.pipelines.registration.registration_icp(
        source,
        target, #前一个点云
        max_correspondence_distance,
        initial_pair,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),  #使用点到面ICP
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations) #设置终止条件
    )

    
    refined_pair = result.transformation    #获取ICP优化矩阵
    refined_global_transform = (refined_global[i - 1] @ refined_pair)   #计算到001坐标系下的全局变换矩阵
    refined_global.append(refined_global_transform)

    #计算ICP相对于初始变换的修正量 = refined_pair
    icp_correction = (refined_pair @ np.linalg.inv(initial_pair))
    #计算平移修正量，提取icp_correction矩阵最后一列
    translation_correction = np.linalg.norm(icp_correction[:3, 3])
    #计算旋转修正角度，提取矩阵左上角3*3
    R = icp_correction[:3, :3]
    cos_angle = (np.trace(R) - 1.0) / 2.0       #根据旋转矩阵的迹计算角度
    cos_angle = np.clip(cos_angle, -1.0, 1.0)   #将结果限制在-1和1之内

    rotation_correction = np.degrees(np.arccos(cos_angle)) #计算反余弦，并将弧度转换为角度

    #打印结果
    print(f"\nCapture {i + 1:03d} -> Capture {i:03d}")

    print(
        f"Before ICP: fitness={before.fitness:.6f}, "
        f"RMSE={before.inlier_rmse:.6f} mm"
    )

    print(
        f"After  ICP: fitness={result.fitness:.6f}, "
        f"RMSE={result.inlier_rmse:.6f} mm"
    )

    print(
        f"Correspondences: {len(result.correspondence_set)}"
    )

    print(
        f"ICP translation correction: "
        f"{translation_correction:.3f} mm"
    )

    print(
        f"ICP rotation correction: "
        f"{rotation_correction:.4f} degrees"
    )

    print(
        "Refined pair transformation (mm):\n",
        refined_pair
    )

    print(
        "Refined global transformation (mm):\n",
        refined_global_transform
    )

    #将当前ICP的pairwise tranformaion矩阵保存为npy文件
    np.save(output_matrix_folder/ f"capture_{i + 1:03d}_to_{i:03d}_icp.npy",refined_pair)


#将全部点云变换到capture001参考系---------------------------
aligned_pcds = []   #保存最终ICP配准后的四组点云

for i, (pcd, transform) in enumerate(zip(pcds, refined_global),start=1):

    #保存当前capture的全局ICP修正矩阵，例：capture003对齐到capture001的ICP修正后的矩阵
    np.save(output_matrix_folder/ f"capture_{i:03d}_icp_global.npy",transform)
    #完整变换矩阵t_total，即经过初始配准、基于平面的修正以及ICP后最终得到的变换阵
    T_total = transform @ plane_global[i - 1]
    np.save(output_matrix_folder/ f"capture_{i:03d}_total.npy",T_total)

    #应用ICP到基于平面修正后的点云
    aligned = copy.deepcopy(pcd)
    aligned.transform(transform)

    aligned_pcds.append(aligned)
    #保存最终点云
    path = output_folder / f"capture_{i:03d}_icp_aligned.ply"

    if not o3d.io.write_point_cloud(str(path), aligned):
        raise IOError(f"Failed to save {path}")

    print(f"Saved: {path}")


#可视化---------------------------------------------------------------
show_alignment(
    aligned_pcds,
    "Point-to-Plane ICP after Plane Correction - Different Capture Colors",
    use_capture_colors=True
)

show_alignment(
    aligned_pcds,
    "Point-to-Plane ICP after Plane Correction - Original RGB Colors"
)

print("\nPoint-to-Plane ICP Fine Registration completed!")
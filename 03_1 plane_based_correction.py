#用于纠正initial registration里旋转矩阵的误差
#用RANSAC分别拟合平面以及对应的法向量，使不同capture的法向量互相平行。
from pathlib import Path
import copy
import csv
import numpy as np
import open3d as o3d

#初始设置 configuration
folder = Path(r"D:\captures_stop_and_go\forward")
sor_folder = folder / "SOR"
initial_folder = folder / "initial_registration_matrices"
output_folder = folder / "plane_based_correction"
output_folder.mkdir(parents=True, exist_ok=True)

marker_ids = [8, 7, 6, 5]  # Same capture ordering as initial registration
marker_x_positions = [0.0, -0.245, -0.545, -0.845]  # meters, informational
DISTANCE_THRESHOLD_MM = 3.0 #点到拟合平面的最大距离阈值为3
RANSAC_N = 3                #每次RANSAC随机选取3个点
NUM_ITERATIONS = 2000       #最大迭代次数20000
VISUALIZE = True
VISUALIZE_RANSAC_EACH_CAPTURE = True
SAVE_CORRECTED_PLY = True
COLORS = [[1, 0, 0], [0, 0.65, 0.93], [1, 0.71, 0], [0.6, 0.2, 0.8]]

#定义RANSAC平面拟合函数
#一个平面：ax+by+cz+d = 0
def fit_plane(pcd):
    #RANSAC返回两个参数：model即平面方程的参数a,b,c,d；indices保存属于平面内的点的index
    #segment_plane()是open3D的平面分割函数
    model, indices = pcd.segment_plane(
        distance_threshold=DISTANCE_THRESHOLD_MM,
        ransac_n=RANSAC_N,
        num_iterations=NUM_ITERATIONS,
    )
    if len(indices) < RANSAC_N:
        raise RuntimeError("RANSAC could not identify a valid plane")
    model = np.asarray(model, dtype=np.float64) #将model保存的平面参数a,b,c,d转换成numpy数组
    norm = np.linalg.norm(model[:3])    #提取前三个参数a,b,c,即为法向量，并计算法向量的欧几里得长度
    if norm < 1e-12:                    #检查法向量是否有效，若无限接近于0则无效
        raise RuntimeError("Invalid plane normal")
    model /= norm     #归一化平面方程，只是让法向量成为单位向量，不会改变平面的方向或位置
    xyz = np.asarray(pcd.points)[indices]   #提取平面内inliers的点的xyz坐标，只包含RANSAC认为的属于拟合平面内的点
    return model, indices, xyz      #返回值：平面参数a,b,c,d，inliers的index，inliers的xyz坐标

#判断平面的法向量是否反向
#因为最终目的不仅是使所有平面的法向量平行，而且必须同向
def orient_plane(model, reference_normal):
    result = model.copy()       #复制平面参数a,b,c,d
    #计算当前平面法向量与参考平面法向量的点积
    #点积>0:同向，=0垂直，<0反向
    if np.dot(result[:3], reference_normal) < 0:    
        result *= -1    #若<0，则所有参数乘-1，构造完全反向的平面 -> 得到方向相反的法向量
    return result   #返回的是同一方向后的平面参数

#计算一个最小的旋转矩阵，使向量a旋转后与向量b平行
#向量b为参考法向量
def minimal_rotation(a, b):
    a = a / np.linalg.norm(a)   #转换为单位向量
    b = b / np.linalg.norm(b)
    cross = np.cross(a, b)      #向量叉积
    s = np.linalg.norm(cross)   #计算叉积长度 s = ||axb||= sinα，叉积为0时则两个向量平行
    c = float(np.clip(np.dot(a, b), -1.0, 1.0)) #计算点积 c = a.b = cosα，>0则两个向量同向，<0则反向
    if s < 1e-12:   #若叉积为0，即两个向量几乎平行
        if c > 0:   #若两个向量同向，则已对齐，不需要旋转
            return np.eye(3)
        #c<0，即两个向量反向
        axis = np.eye(3)[np.argmin(np.abs(a))]  #选择一个尽量与a不平行的坐标轴
        axis = np.cross(a, axis)                #计算一条垂直于a的单位旋转轴
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)       #使用180度旋转公式，Rodrigues公式在旋转角度为180度时的特殊形式
    #一般情况，a和b既不平行也不反向时，用Rodrigues旋转公式
    K = np.array([[0, -cross[2], cross[1]],
                  [cross[2], 0, -cross[0]],
                  [-cross[1], cross[0], 0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))

#函数：评估一个点云拟合平面与参考平面之间的几何一致性
#输入：model拟合平面的参数a,b,c,d
#inlier_xyz当前点云中属于拟合平面的内点坐标
#reference：参考点云的拟合平面参数ar,br,cr,dr
def plane_metrics(model, inlier_xyz, reference):
    n = model[:3]
    nr, dr = reference[:3], reference[3]
    angle = np.degrees(np.arccos(np.clip(abs(np.dot(n, nr)), 0, 1)))
    own = inlier_xyz @ n + model[3]
    relative = inlier_xyz @ nr + dr
    return {
        "normal_angle_deg": float(angle),       #两个平面间的法向量夹角
        "signed_mean_distance_mm": float(relative.mean()),  #当前平面内点到参考平面的平均有符号距离
        "reference_plane_rmse_mm": float(np.sqrt(np.mean(relative**2))),    #当前平面内点到参考平面的均方根距离
        "own_plane_rmse_mm": float(np.sqrt(np.mean(own**2))),   #当前平面内点到自身拟合平面的均方根距离
    }


#open3D可视化点云--------------------------------------------------------
#将多个点云显示在同一个窗口中
#参数：clouds需要可视化的点云；title窗口标题；original_rgb是否保留点云原始rgb颜色，默认false
def show_clouds(clouds, title, original_rgb=False):
    display = []
    for pcd, color in zip(clouds, COLORS):
        c = copy.deepcopy(pcd)

        if not original_rgb:
            c.paint_uniform_color(color)

        display.append(c)

    o3d.visualization.draw_geometries(
        display,
        window_name=title,
        width=1280,
        height=720
    )


#用SOR后的点云做初配准，或许可以直接用初配准后的结果
initial_clouds = []
initial_transforms = []
for capture in range(1, 5):
    ply_path = sor_folder / f"capture_{capture:03d}_sor.ply"
    matrix_path = initial_folder / f"capture_{capture:03d}_initial.npy"
    if not ply_path.is_file():
        raise FileNotFoundError(f"Missing preprocessed cloud: {ply_path}")
    if not matrix_path.is_file():
        raise FileNotFoundError(f"Missing initial transform: {matrix_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        raise ValueError(f"Empty point cloud: {ply_path}")
    T = np.load(matrix_path).astype(np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f"Invalid initial transform: {matrix_path}")
    if not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError(f"Invalid homogeneous matrix last row: {matrix_path}")
    if capture == 1 and not np.allclose(T, np.eye(4), atol=1e-6):
        raise ValueError("Capture 001 must be the global reference (identity transform)")
    pcd.transform(T)
    initial_clouds.append(pcd)
    initial_transforms.append(T)
    print(f"Loaded Capture {capture:03d} (Marker {marker_ids[capture-1]}): {len(pcd.points)} points")

#用RANSAC拟合capture001的平面，用作参考平面---------------------------------------------
#用上面自定义的fit_plane()函数
reference_model, _, _ = fit_plane(initial_clouds[0])
if reference_model[2] < 0:   #若z分量为负，则翻转整个平面
    reference_model *= -1
print("\nReference plane Capture 001 [a, b, c, d]:", reference_model)
print("Plane equation: ax + by + cz + d = 0; XYZ and d are in mm")

if VISUALIZE:
    show_clouds(initial_clouds, "Before Plane-Based Correction")  #调用自定义的show_clouds()函数

#修正capture002-004的旋转
corrected_clouds = []   #修正后的点云
evaluation_rows = []    #每组点云修正前后的评价指标
plane_rows = []
for i, initial_pcd in enumerate(initial_clouds):    #遍历4组initial后的点云
    capture = i + 1
    #计算修正前的平面参数
    before_model, _, before_xyz = fit_plane(initial_pcd)                #RANSAC平面拟合
    before_model = orient_plane(before_model, reference_model[:3])      #统一法向量方向
    before = plane_metrics(before_model, before_xyz, reference_model)   #计算修正前的Evaluation指标

    T_correction = np.eye(4, dtype=np.float64)  #创建4*4的单位修正矩阵
    if i > 0: #0对应的capture001，表示只修正002，003，004
        R = minimal_rotation(before_model[:3], reference_model[:3]) #统一当前法向量的方向与参考平面法向量的方向
        centroid = before_xyz.mean(axis=0)  #计算当前平面内点的质心，后续围绕这个位置进行旋转
        #计算沿参考法向量方向的平移，使最终的平面之间不仅平行，且在同一平面上
        normal_shift = -(np.dot(reference_model[:3], centroid) + reference_model[3])
        #绕质心旋转所需的平移补偿 + 法向平移修正？
        t = centroid - R @ centroid + normal_shift * reference_model[:3]
        T_correction[:3, :3] = R
        T_correction[:3, 3] = t

    corrected = copy.deepcopy(initial_pcd)  #复制初配准后的点云
    corrected.transform(T_correction)       #应用上面得到的修正矩阵T——correction
    corrected_clouds.append(corrected)      #保存修正后的点云

    #对修正后的点云重新进行RANSAC拟合
    after_model, after_indices, after_xyz = fit_plane(corrected)
    after_model = orient_plane(after_model, reference_model[:3])
    after = plane_metrics(after_model, after_xyz, reference_model)
    ratio = len(after_indices) / len(corrected.points)

    #计算最终配准矩阵
    T_final = T_correction @ initial_transforms[i]
    np.save(output_folder / f"capture_{capture:03d}_plane_correction.npy", T_correction)
    np.save(output_folder / f"capture_{capture:03d}_final.npy", T_final)
    if SAVE_CORRECTED_PLY:
        o3d.io.write_point_cloud(
            str(output_folder / f"capture_{capture:03d}_plane_corrected.ply"), corrected
        )

    #保存评价指标
    evaluation_rows.append({
        "capture": capture, "marker_id": marker_ids[i],
        **{f"before_{k}": v for k, v in before.items()},
        **{f"after_{k}": v for k, v in after.items()},
    })
    plane_rows.append({
        "capture": capture,
        "marker_id": marker_ids[i],
        "a": float(after_model[0]), "b": float(after_model[1]),
        "c": float(after_model[2]), "d_mm": float(after_model[3]),
        "inlier_count": len(after_indices), "inlier_ratio": ratio,
        **after,
    })
    print(f"\n--- Capture {capture:03d}, Marker {marker_ids[i]} ---")
    print("Correction matrix (mm):\n", T_correction)
    print("Final matrix (mm):\n", T_final)
    print("Post-correction RANSAC plane [a,b,c,d]:", after_model)
    print("Post-correction unit normal:", after_model[:3])
    print(f"RANSAC inlier ratio: {ratio:.2%}")
    for metric in before:
        print(f"{metric}: {before[metric]:.5f} -> {after[metric]:.5f}")

    if VISUALIZE and VISUALIZE_RANSAC_EACH_CAPTURE:
        plane_inliers = corrected.select_by_index(after_indices)
        remaining = corrected.select_by_index(after_indices, invert=True)
        plane_inliers.paint_uniform_color([0, 0.8, 0.1])
        remaining.paint_uniform_color([0.9, 0.15, 0.15])
        o3d.visualization.draw_geometries(
            [plane_inliers, remaining],
            window_name=f"Capture {capture:03d}: RANSAC inliers (green) / others (red)"
        )

for filename, rows in [
    ("plane_correction_evaluation.csv", evaluation_rows),
    ("post_correction_ransac_planes.csv", plane_rows),
]:
    with (output_folder / filename).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

#Visualization after Plane-Based Correction --------------------------
#可视化修正后的点云-----------------------------------------------------

if VISUALIZE:
    #corrected point clouds with different colors
    #染色的点云
    show_clouds(
        corrected_clouds,
        "After Plane-Based Correction - Colored",
        original_rgb=False
    )

    #corrected point clouds with original RGB colors
    #原RGB颜色的点云
    show_clouds(
        corrected_clouds,
        "After Plane-Based Correction - Original RGB",
        original_rgb=True
    )

print("\nFinished. Outputs:", output_folder)
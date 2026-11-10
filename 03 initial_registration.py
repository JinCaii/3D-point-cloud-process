import open3d as o3d
import numpy as np
import copy
from pathlib import Path

#辅助可视化函数---------------------------------------------------------------------------------
#3个参数：源点云、目标点云、4*4变换阵
#变换阵通过marker detection的位姿估计得到
def draw_registration_result(source, target, transformation):
    #创建源点云和目标点云的副本，因为后面要染色，直接在原始点云上改的话会修改原始数据
    #用pythond的copy.deepcopy复制，所以一开始要import copy
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    #给点云设置颜色
    source_temp.paint_uniform_color([1, 0.706, 0])        #source: 黄色
    target_temp.paint_uniform_color([0, 0.651, 0.929])    #target: 青色
    #将源点云副本变换到目标点云的坐标系Initial Registration
    #只变换源点云，不变换目标点云
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

#初始配准------------------------------------------------------------------------------------

#文件路径
folder = Path(r"D:\captures_stop_and_go\forward")
preprocessed_folder = folder / "SOR"
matrix_folder = folder / "transformation_matrices"

#读取4个预处理后的点云
pcds = []
for i in range(1, 5):
    ply_path = preprocessed_folder / f"capture_{i:03d}_sor.ply"
    pcd = o3d.io.read_point_cloud(str(ply_path))

    if pcd.is_empty():
        raise ValueError(f"Point cloud is empty: {ply_path}")

    pcds.append(pcd)
    print(f"Loaded: {ply_path.name}")
    print(f"Number of points: {len(pcd.points)}")


#定义marker id及其在统一参考坐标系中的位置
#以marker 8为参考，沿-X方向排列
#单位：m，与marker detection保存的位姿矩阵一致
marker_ids = [8, 7, 6, 5]
marker_x_positions = [0.0, -0.245, -0.545, -0.845]  #在工件上测量的，已知marker之间的间距


#读取marker Detection得到的4*4变换矩阵
#这些矩阵表示marker -> camera，平移单位为m
marker_matrices = []
for i, marker_id in enumerate(marker_ids, start=1):
    matrix_path = (matrix_folder / f"capture_{i:03d}_marker_{marker_id}.npy")
    T_marker_to_camera = np.load(matrix_path)

    if T_marker_to_camera.shape != (4, 4):
        raise ValueError(f"Invalid matrix shape: {matrix_path}")

    marker_matrices.append(T_marker_to_camera)
    print(f"\nLoaded transformation matrix: {matrix_path.name}")
    print(T_marker_to_camera)


#建立Global Reference Coordinate System
#选择Capture 001作为全局参考点云，因此Capture 001不需要变换，使用单位矩阵
T_C1_M8 = marker_matrices[0]   #T_C1_M8指Capture1，marker8的变换阵
global_transforms = [np.eye(4, dtype=np.float64)]

#计算Capture 002、003、004到Capture 001的初始变换矩阵
for i in range(1, 4):

    #构建当前marker相对于marker8的变换矩阵
    #所有marker方向一致，相对于marker8只存在x负方向的位移
    T_M8_Mi = np.eye(4, dtype=np.float64)
    T_M8_Mi[0, 3] = marker_x_positions[i]

    #计算source -> Global Reference的变换矩阵
    #T_C1_Ci = T_C1_M8 @ T_M8_Mi @ inv(T_Ci_Mi)
    T_source_to_global = (
        T_C1_M8
        @ T_M8_Mi
        @ np.linalg.inv(marker_matrices[i])
    )

    #Marker位姿矩阵的平移单位为m
    #Zivid PLY点云坐标单位为mm，因此只转换平移部分
    T_source_to_global[:3, 3] *= 1000.0

    # 保存当前变换矩阵
    global_transforms.append(T_source_to_global)

    print(f"\nInitial transformation: Capture {i+1:03d} -> Capture 001")
    print(T_source_to_global)

#可视化Initial Registration结果
#Capture 001作为Target，其余三个依次作为Source
for i in range(1, 4):

    print(f"\nVisualizing Capture {i+1:03d} -> Capture 001")

    draw_registration_result(
        pcds[i],                # Source
        pcds[0],                # Target
        global_transforms[i]    # Initial Transformation
    )

#将四个点云全部变换到Global Reference Coordinate System
aligned_pcds = []
for i in range(4):

    #使用副本，避免修改原始点云
    aligned_pcd = copy.deepcopy(pcds[i])

    #应用初始变换矩阵
    aligned_pcd.transform(global_transforms[i])
    aligned_pcds.append(aligned_pcd)

#可视化四个初始配准后的点云
#给不同Capture设置不同颜色，便于观察重叠情况
colors = [
    [1, 0, 0],       # Capture 001：红色
    [0, 1, 0],       # Capture 002：绿色
    [0, 0, 1],       # Capture 003：蓝色
    [1, 0.706, 0]    # Capture 004：黄色
]

visualization_pcds = []

for i in range(4):

    pcd_temp = copy.deepcopy(aligned_pcds[i])
    pcd_temp.paint_uniform_color(colors[i])

    visualization_pcds.append(pcd_temp)

print("\nVisualizing all initially aligned point clouds...")

o3d.visualization.draw_geometries(
    visualization_pcds,
    window_name="Initial Registration - All Point Clouds - Different Colors"
)

#可视化四个初始配准后的点云（保留原始RGB颜色）
print("\nVisualizing initially aligned point clouds with original colors...")

o3d.visualization.draw_geometries(
    aligned_pcds,
    window_name="Initial Registration - Original Colors"
)

print("\nInitial Registration completed!")

#初始配准效果Evaluation --------------------------------------------------------

print("\nInitial Registration Evaluation")

#最大对应点距离，单位 mm（PLY点云使用mm）
threshold = 10.0

#依次评估相邻点云
for i in range(1, 4):

    # Source：当前Capture
    source = pcds[i]

    # Target：前一个Capture
    target = pcds[i - 1]

    # 计算Source -> Target的相对变换矩阵
    T_source_to_target = (
        np.linalg.inv(global_transforms[i - 1])
        @ global_transforms[i]
    )

    # 保存Initial Registration的全局变换矩阵
    initial_matrix_folder = folder / "initial_registration_matrices"
    initial_matrix_folder.mkdir(parents=True, exist_ok=True)

    for i, T in enumerate(global_transforms, start=1):
        np.save(
            initial_matrix_folder / f"capture_{i:03d}_initial.npy",
            T
        )

    # Evaluation
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source,
        target,
        threshold,
        T_source_to_target
    )

    print(f"\nCapture {i+1:03d} -> Capture {i:03d}")
    print(f"Fitness: {evaluation.fitness:.6f}")
    print(f"Inlier RMSE: {evaluation.inlier_rmse:.6f} mm")
    print(f"Correspondences: {len(evaluation.correspondence_set)}")

print("\nInitial Registration Evaluation completed!")
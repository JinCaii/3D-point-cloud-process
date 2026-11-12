import open3d as o3d
import numpy as np
import copy
from pathlib import Path

#辅助可视化函数---------------------------------------------------------------------------------
#Helper visualization function-----------------------------------------------------------------
#3个参数：源点云、目标点云、4*4变换阵
#变换阵通过marker detection的位姿估计得到
#3 parameters: source point cloud, target point cloud, and a 4×4 transformation matrix
#the transformation matrix is obtained from pose estimation using 01 marker_detection.py
def draw_registration_result(source, target, transformation):
    #创建源点云和目标点云的副本，因为后面要染色，直接在原始点云上改的话会修改原始数据
    #用python的copy.deepcopy复制，所以一开始要import copy
    #create copies of the source and target point clouds because the original data would be modified if colors were applied directly
    #use Python's copy.deepcopy() to create copies, so copy must be imported at the beginning
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    #给点云设置颜色
    #assign colors to the point clouds
    source_temp.paint_uniform_color([1, 0.706, 0])        #source: 黄色 #source:yellow
    target_temp.paint_uniform_color([0, 0.651, 0.929])    #target: 青色 #target:cyan
    #将源点云副本变换到目标点云的坐标系Initial Registration
    #只变换源点云，不变换目标点云
    #transform the source point cloud copy into the target point cloud coordinate system for initial registration
    #only transform the source point cloud, leaving the target point cloud unchanged
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

#初始配准------------------------------------------------------------------------------------
#initial registration-----------------------------------------------------------------------

#文件路径
#file path
folder = Path(r"D:\captures_stop_and_go\forward")
preprocessed_folder = folder / "SOR"                #预处理后的点云     #preprocessed point clouds
matrix_folder = folder / "transformation_matrices"  #marker detection得到的marker相对于相机的位姿 #marker poses relative to the camera obtained from marker detection

#读取4个预处理后的点云
#load 4 preprocessd point clouds
pcds = []
for i in range(1, 5):
    ply_path = preprocessed_folder / f"capture_{i:03d}_sor.ply"
    pcd = o3d.io.read_point_cloud(str(ply_path))

    if pcd.is_empty():
        raise ValueError(f"Point cloud is empty: {ply_path}")

    pcds.append(pcd)       #将当前点云对象pcd保存到列表pcds中  #append the current point cloud object pcd to the list pcds
    print(f"Loaded: {ply_path.name}")
    print(f"Number of points: {len(pcd.points)}")


#定义marker id及其在统一参考坐标系中的位置
#以marker 8为参考，沿-X方向排列
#单位：m，与marker detection保存的位姿矩阵一致
#define the marker id and their positions in a common reference coordinate system.
#use marker 8 as the reference, with the other markers arranged along the negative X-axis
#unit: m, consistent with the pose matrices saved during marker detection

marker_ids = [8, 7, 6, 5]
marker_x_positions = [0.0, -0.245, -0.545, -0.845]  
#在工件上测量的，已知marker之间的间距，用于后续把capture2，3，4变换到captrue1的坐标系
#measured on the workpiece. The known distances between markers are used to transform Captures 002, 003, and 004 into the coordinate system of Capture 001


#读取marker Detection得到的4*4变换矩阵
#这些矩阵表示marker -> camera，平移单位为m
#load the 4×4 transformation matrices obtained from marker detection
#these matrices represent marker -> camera transformations, unit m
marker_matrices = []
for i, marker_id in enumerate(marker_ids, start=1): #遍历所有marker #iterate through all markers
    matrix_path = (matrix_folder / f"capture_{i:03d}_marker_{marker_id}.npy")
    T_marker_to_camera = np.load(matrix_path)

    if T_marker_to_camera.shape != (4, 4):
        raise ValueError(f"Invalid matrix shape: {matrix_path}")

    marker_matrices.append(T_marker_to_camera)
    print(f"\nLoaded transformation matrix: {matrix_path.name}")
    print(T_marker_to_camera)


#建立全局参考坐标系global reference coordinate system
#选择Capture 001作为全局参考点云，因此Capture 001不需要变换，使用单位矩阵
#establish the global reference coordinate system
#select Capture 001 as the global reference point cloud. Therefore Capture 001 does not require transformation, and the identity matrix is used


T_C1_M8 = marker_matrices[0]   #T_C1_M8指Capture1，marker8的变换阵 #T_C1_M8 represents the transformation matrix of marker 8 in Capture 001
global_transforms = [np.eye(4, dtype=np.float64)]   #创建单位阵，capture001不需要变换 #create an identity matrix because Capture 001 does not require transformation

#计算Capture 002、003、004到Capture 001的初始变换矩阵T_source_to_global
#这里是直接计算004到001的变换阵，而不是004align到003，再align到002，再align到001，以减少累积误差
#calculate the initial transformation matrices T_source_to_global from Captures 002, 003, and 004 to Capture 001
#the transformation from Capture 004 to Capture 001 is calculated directly, rather than sequentially aligning 004 to 003, 003 to 002, and 002 to 001, to reduce accumulated errors
for i in range(1, 4):   #遍历剩下3个capture #iterate through the remaining three captures
    #构建当前marker相对于marker8的变换矩阵  #construct the transformation matrix of the current marker relative to marker 8
    #所有marker方向一致，相对于marker8只存在x负方向的位移 #all markers have the same orientation, so only translation along the negative X-axis exists relative to marker 8
    T_M8_Mi = np.eye(4, dtype=np.float64)
    T_M8_Mi[0, 3] = marker_x_positions[i]

    #计算source -> 全局坐标系的变换矩阵
    #calculate the transformation matrix from source to the global coordinate system
    T_source_to_global = (
        T_C1_M8
        @ T_M8_Mi  #@表示矩阵乘法  #@ is matrix multiplication
        @ np.linalg.inv(marker_matrices[i]) #逆矩阵，得到相机相对于marker的位姿 #invert the matrix to get the camera pose relative to the marker
    )

    #marker位姿矩阵的平移单位为m，而zivid ply的单位是mm，这里将m转换成mm
    #[:3, 3]中:3转换第0，1，2行，第二个3表示第3列，即只转换平移部分
    #the translation components of the marker pose matrices are in meters, while Zivid PLY point clouds use millimeters. Convert meters to millimeters here
    #[:3, 3], :3 selects rows 0, 1, and 2, while the second 3 selects column 3. i.e. only the translation components are converted
    T_source_to_global[:3, 3] *= 1000.0  

    #保存当前变换矩阵到列表global_transforms
    #append the current transformation matrix to the global_transforms list
    global_transforms.append(T_source_to_global)

    print(f"\nInitial transformation: Capture {i+1:03d} -> Capture 001")
    print(T_source_to_global)

#可视化初始配准的结果
#capture 001作为target，其余三个依次作为source
#visualize the initial registration results
#use Capture 001 as the target and the other three captures as sources sequentially
for i in range(1, 4):

    print(f"\nVisualizing Capture {i+1:03d} -> Capture 001")

    draw_registration_result(
        pcds[i],                #source point cloud
        pcds[0],                #target
        global_transforms[i]    #initial tranformation matrix
    )

#将四个点云全部变换到全局坐标系
#transform all 4 point clouds into the global coordinate system
aligned_pcds = []   #保存初始配准后的点云 #store the initially registrated point clouds
for i in range(4):
    #使用副本，避免修改原始点云
    #use a copy to avoid modifying the original point cloud
    aligned_pcd = copy.deepcopy(pcds[i])
    #配准！！！！！
    #REGISTRATION!!!!!!!!!
    aligned_pcd.transform(global_transforms[i])
    aligned_pcds.append(aligned_pcd)

#可视化四个初始配准后的点云
#给不同Capture设置不同颜色，便于观察重叠情况
#visualize the four initially registered point clouds
#sssign different colors to different captures to make the overlapping regions easier to observe
colors = [
    [1, 0, 0],       #capture 001：红色 red
    [0, 1, 0],       #capture 002：绿色 green
    [0, 0, 1],       #capture 003：蓝色 blue
    [1, 0.706, 0]    #capture 004：黄色 yellow
]

visualization_pcds = [] #创建列表，保存可视化的点云 #create list to store the point clouds for visualization

for i in range(4):
    pcd_temp = copy.deepcopy(aligned_pcds[i])   #复制第i个配准后的点云 copy the i-th registered point cloud
    pcd_temp.paint_uniform_color(colors[i])     #上色 coloring
    visualization_pcds.append(pcd_temp)

print("\nVisualizing all initially aligned point clouds")
o3d.visualization.draw_geometries(
    visualization_pcds,
    window_name="Initial Registration - All Point Clouds - Different Colors"
)

#可视化四个初始配准后的点云（原始RGB颜色）
#visualize the four initially registered point clouds with their original RGB colors
print("\nVisualizing initially aligned point clouds with original colors...")
o3d.visualization.draw_geometries(
    aligned_pcds,
    window_name="Initial Registration - Original Colors"
)

print("\nInitial Registration completed!")

#初始配准效果Evaluation --------------------------------------------------------
#Initial Registration Evaluation-----------------------------------------------
print("\nInitial Registration Evaluation")
#最大对应点距离，单位 mm（ply点云使用mm）
#maximum correspondence distance in mm
threshold = 10.0

#依次评估相邻点云
#evaluate adjacent point clouds sequentially
for i in range(1, 4):
    source = pcds[i]        #source：当前capture  source: current capture
    target = pcds[i - 1]    #target：前一个capture  target: a previous capture
    #即用004与003，003与002，002与001对比，而不是004与001对比
    #this means comparing 004 with 003, 003 with 002, and 002 with 001, rather than comparing 004 directly with 001

    #计算source -> target的相对变换矩阵
    #使用的是原始坐标下的点云 pcds，而不是已经变换到全局坐标系的 aligned_pcds，所以需要求相邻source->target的目标矩阵
    #calculate the relative transformation matrix source -> target
    #the original point clouds pcds are used instead of aligned_pcds, which have already been transformed into the global coordinate system 
    #therefore,= the relative source->target transformation matrix between adjacent captures must be calculated



    T_source_to_target = (  #求的是capture i相对于capture(i-1)的位姿 #calculate the pose of Capture i relative to Capture (i-1)
        np.linalg.inv(global_transforms[i - 1]) #capture(i-1)相对于Capture1的位姿的逆 #inverse of the pose of Capture (i-1) relative to Capture 001
        @ global_transforms[i]                  #capture i相对于capture1的位姿 #pose of Capture i relative to Capture 001
    )

    #保存初始配准的全局变换矩阵-----------------------------------------------------------
    #save the global transformation matrices from initial registration ----------------------------------
    #创建文件夹
    #create a folder
    initial_matrix_folder = folder / "initial_registration_matrices"
    initial_matrix_folder.mkdir(parents=True, exist_ok=True)

    #从i=1，即capture002开始遍历列表global_transforms里的全局变换阵
    #iterate through the global transformation matrices in global_transforms, starting with i=1, i.e. Capture 002
    for i, T in enumerate(global_transforms, start=1):  
        np.save(    #保存成.npy文件 #save as a .npy file
            initial_matrix_folder / f"capture_{i:03d}_initial.npy",
            T
        )

    #函数evaluation有四个参数：source,target,threshold,变换矩阵
    #输出两个指标Fitness以及RMSE
    #fitness:如0.8指源点云中有80%的点能在目标点云中找到距离不超过threshold的点，即80%的源点找到了对应点
    #fitness越接近1越好，但是它也会受到点云重叠率以及threshold的影响
    #inlier RMSE：经过变换的源点于对应目标点之间的距离误差
    #RMSE越小，则inlier对应点之间的对齐程度越高
    #the evaluation function takes four parameters: source, target, threshold, and transformation matrix
    #it outputs two evaluation metrics: Fitness and RMSE
    #fitness: For example, a value of 0.8 means that 80% of the source points have corresponding points in the target point cloud within the specified threshold
    #a Fitness value closer to 1 is better, but it is also affected by the point cloud overlap ratio and the threshold
    #inlier RMSE: The distance error between the transformed source points and their corresponding target points
    #a lower RMSE indicates better alignment between the inlier correspondences
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, T_source_to_target)

    print(f"\nCapture {i+1:03d} -> Capture {i:03d}")
    print(f"Fitness: {evaluation.fitness:.6f}")
    print(f"Inlier RMSE: {evaluation.inlier_rmse:.6f} mm")
    print(f"Correspondences: {len(evaluation.correspondence_set)}")

print("\nInitial registration evaluation completed!")
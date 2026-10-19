import open3d as o3d

# 获取 Open3D 官方示例点云，并指定保存位置
data = o3d.data.DemoCropPointCloud(
    data_root=r"C:\Users\Jin Cai\Desktop\open3D\data"
)

# 读取点云
pcd = o3d.io.read_point_cloud(data.point_cloud_path)

# 查看信息
print(pcd)
print("点云文件位置:", data.point_cloud_path)

# 显示点云
o3d.visualization.draw_geometries([pcd])
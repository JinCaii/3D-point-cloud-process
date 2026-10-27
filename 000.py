import zivid

print("CameraIntrinsics:", hasattr(zivid, "CameraIntrinsics"))

if hasattr(zivid, "CameraIntrinsics"):
    print("load:", hasattr(zivid.CameraIntrinsics, "load"))
    print("相关方法:", [
        name for name in dir(zivid.CameraIntrinsics)
        if "load" in name.lower()
        or "file" in name.lower()
        or "init" in name.lower()
    ])


# 读取之前保存的相机内参
with open(folder / "intrinsics_2D.yml", "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

# 提取相机内参
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
import zivid
from pathlib import Path

# 选择一个读取失败的 ZDF 文件
zdf_path = Path(
    r"C:\Users\User\Desktop\captures_continuousMotion\keyframe\capture_003.zdf"
)

print("文件是否存在:", zdf_path.exists())

if zdf_path.exists():
    print("文件大小:", zdf_path.stat().st_size, "bytes")

try:
    # 初始化 Zivid SDK
    with zivid.Application() as app:

        # 加载 ZDF 文件
        frame = zivid.Frame(str(zdf_path))
        print("ZDF 加载成功！")

        # 获取点云
        point_cloud = frame.point_cloud()
        print("点云读取成功！")

        # 提取 RGBA
        rgba = point_cloud.copy_data("rgba")
        print("RGBA 提取成功！")
        print("RGBA shape:", rgba.shape)

except Exception as error:
    print("读取失败:", repr(error))
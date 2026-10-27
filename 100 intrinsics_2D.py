#连接相机前要先在studio断开连接
import zivid
import zivid.experimental.calibration as calibration  #读取相机内参

with zivid.Application() as app:
    print("Connecting to Zivid camera...")
    with app.connect_camera() as camera:
        print(f"Connected: {camera.info}")

        #加载从 Zivid Studio 导出的拍摄参数
        settings = zivid.Settings2D.load(
            r"C:\Users\Jin Cai\Desktop\captures_stop_and_go\settings2D.yml"
        )

        #获取内参
        intrinsics = calibration.intrinsics(camera, settings)

        #保存为yml文件
        intrinsics.save(r"C:\Users\Jin Cai\Desktop\captures_stop_and_go\intrinsics_2D.yml")

        print("Camera intrinsics saved!")
#连接相机前要先在studio断开连接
import zivid
import zivid.experimental.calibration as calibration  #读取相机内参

with zivid.Application() as app:
    print("Connecting to Zivid camera...")
    with app.connect_camera() as camera:
        print(f"Connected: {camera.info}")
        
        #加载从 Zivid Studio 导出的拍摄参数
        settings = zivid.Settings.load(
            r"C:\Users\Jin Cai\Desktop\captures_continuousMotion\settings.yml"
        )

        #使用该设置拍摄一次
        frame = camera.capture(settings)
        
        #根据拍摄结果估计相机内参
        intrinsics = calibration.estimate_intrinsics(frame)

        #保存为yml文件
        intrinsics.save(r"C:\Users\Jin Cai\Desktop\captures_continuousMotion\intrinsics_3d.yml")

        print("Camera intrinsics saved!")
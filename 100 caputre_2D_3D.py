#用于stop-and-go中分别采集2D和3D图像
#zivid studio不支持按一次capture按钮而自动采集2D和3D图像，需要手动切换
#该程序实现了按一下enter自动采集一次2D和3D图像，无需手动切换
#采集时保存相机内参，用于后续aruco相机位姿估计

#连接相机前要先在studio断开连接

from pathlib import Path
import zivid
import zivid.experimental.calibration #读取相机内参

#创建保存文件夹
output = Path(r"C:\Users\Jin Cai\Desktop\captures_stop_and_go")
output.mkdir(parents=True, exist_ok=True)

#连接zivid相机
app = zivid.Application()
print("Connecting to Zivid camera...")
camera = app.connect_camera()
print(f"Connected: {camera.info}")

#加载独立的2D和3D拍摄参数
settings_2d = zivid.Settings2D.load( r"C:\Users\Jin Cai\Desktop\captures_stop_and_go\settings2D.yml") #自己更改路径
settings_3d = zivid.Settings.load( r"C:\Users\Jin Cai\Desktop\captures_stop_and_go\settings3D.yml") #自己更改路径

#获取并保存2D相机内参，用于后续arUco位姿估计
#输出是intrinsics_2d.yml，和拍摄的图像保存在同一个文件夹中
# intrinsics_2d = zivid.experimental.calibration.intrinsics(
#     camera, settings_2d
# )
# intrinsics_2d.save(str(output / "intrinsics_2d.yml"))
# print("2D相机内参已保存！")

# # 获取并保存3D拍摄对应的相机内参
# intrinsics_3d = zivid.experimental.calibration.intrinsics(
#     camera, settings_3d
# )
# intrinsics_3d.save(str(output / "intrinsics_3d.yml"))
# print("3D拍摄对应的相机内参已保存！")



#拍摄
count = 0 #初始化计数器

while True:
    command = input("按Enter拍摄，输入q退出：") #等待用户输入

    if command.lower() == "q": #lower可以将大写字母转换成小写字母，考虑到了开启大写锁定的情况
        break

    count += 1 #每按一次enter，计数器+1

    # 独立采集2D图像，不是从3D图像中获取2D图像
    frame_2d = camera.capture(settings_2d) #用提前设置的2D参数
    image = frame_2d.image_rgba()
    image.save(str(output / f"capture_{count:03d}.png"))  #文件名capture_001.png

    #设置投影仪亮度为0，则切换之间无需等待

    # 独立采集3D点云
    frame_3d = camera.capture(settings_3d) #用提前设置的3D拍摄参数
    frame_3d.save(str(output / f"capture_{count:03d}.zdf"))  #文件名capture_001.zdf

    print(f"第{count}次拍摄完成！")

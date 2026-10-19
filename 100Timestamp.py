#每隔1s自动拍照
#用于从3D中读取2D信息
import zivid
import time
from datetime import datetime
from pathlib import Path


# 参数设置
INTERVAL = 1  # 采集启动间隔，单位：秒
SAVE_DIR = Path(r"C:\Users\User\Desktop\zivid_captures")

# 创建保存文件夹
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# 初始化 Zivid 相机
app = zivid.Application()
camera = app.connect_camera()

# 加载在Zivid Studio中导出的采集设置文件yaml
# 修改YAML路径
settings = zivid.Settings.load(
    r"C:\Users\User\Desktop\3Dsettings.yml"
)

print("Zivid camera connected!")
print("Start capturing every 1s")
#Press Ctrl+C to stop


# 定时采集
count = 0
next_capture = time.perf_counter()

try:#尝试执行的代码
    while True:

        # 等待下一个计划采集时刻
        #next_capture计划下一次开始的时间
        wait_time = next_capture - time.perf_counter()  #time.perf_counter()读取当前时间，单增计时器

        if wait_time > 0:  #若下一次采集时间在当前之后
            time.sleep(wait_time) #等待wait_time后再执行，如下次采集与当前时间间隔5ms，就等待5ms

        # 记录开始时间
        start = time.perf_counter() #记录的是计时器的时间
        timestamp = datetime.now().isoformat(timespec="milliseconds") #获取当前电脑时间，用于和encoder同步的时间戳

        # 采集 3D 点云
        frame = camera.capture(settings)

        # 记录采集调用耗时
        capture_duration = time.perf_counter() - start #当前计时器时间-开始计时器时间

        # 保存为 zdf文件
        zdf_filename = SAVE_DIR / f"{count:03d}.zdf"#保存的文件名为3位，不足的地方用0补齐，如000，001，002
        frame.save(str(zdf_filename))
         
        

        #保存为ply文件
        ply_filename = SAVE_DIR / f"{count:03d}.ply"
        frame.save(str(ply_filename))


        print(
        f"Frame {count:03d} | "
        f"Start: {timestamp} | "
        f"Capture call: {capture_duration:.3f} s | "
        f"Saved: {zdf_filename.name}, {ply_filename.name}"
        )
        count += 1 #文件编号+1

        #计算下一次采集时间
        next_capture += INTERVAL
        now = time.perf_counter()

        while next_capture < now:
            next_capture += INTERVAL

except KeyboardInterrupt:#键盘中断
    print("\nCapture stopped.")

finally:
    print(f"Total frames captured: {count}")

import zivid
import time
import numpy as np

# 参数
TEST_COUNT = 30

SETTINGS_PATH = (
    r"C:\Users\Jin Cai\Desktop"
    r"\captures_continuousMotion\settings.yml"
)

# 初始化相机
app = zivid.Application()
print("Connecting to Zivid camera...")
camera = app.connect_camera()
print("connected")

settings = zivid.Settings.load(SETTINGS_PATH)

capture_times = []

print("Start capture time test")

for i in range(TEST_COUNT):

    start = time.perf_counter()

    frame = camera.capture(settings)

    capture_time = time.perf_counter() - start

    capture_times.append(capture_time)

    print(
        f"Frame {i:03d} | "
        f"Capture Time: {capture_time:.3f} s"
    )

# 统计结果
print("\nResults")

print(f"Average: {np.mean(capture_times):.3f} s")
print(f"Minimum: {np.min(capture_times):.3f} s")
print(f"Maximum: {np.max(capture_times):.3f} s")
print(f"Median:  {np.median(capture_times):.3f} s")
print(f"P95:     {np.percentile(capture_times, 95):.3f} s")
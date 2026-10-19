import time

interval = 0.2  # 200 ms

while True:
    start = time.perf_counter()

    frame = camera.capture(settings)

    elapsed = time.perf_counter() - start
    remaining = interval - elapsed

    if remaining > 0:
        time.sleep(remaining)
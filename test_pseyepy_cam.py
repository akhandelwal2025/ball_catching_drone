from pseyepy import Camera, Stream, cam_count
import cv2
import time

c = Camera(fps=150, 
           resolution=Camera.RES_SMALL, 
           colour=True, 
           auto_gain = False, 
           auto_exposure = False,
           gain=63, 
           exposure=255)
# c.check_fps(n_seconds=60)

num_seconds = 10
start = time.time()
count = 0
try:
    while True:
        frame, timestep = c.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        cv2.imshow("120 Hz image", frame)
        cv2.waitKey(1)
        count += 1
except:
    seconds_elapsed = time.time() - start
    print(f"frames recorded in {seconds_elapsed} sec: {count}| frames per second: {count  / seconds_elapsed}")

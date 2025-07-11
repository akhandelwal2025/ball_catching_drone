from pseyepy import Camera, Stream, cam_count
import cv2
import time
from src.vis import Vis

c = Camera(fps=150, 
           resolution=Camera.RES_SMALL,
           gain=63,
           exposure=255)
n_cams = cam_count()
# c.check_fps(n_seconds=60)

num_seconds = 10
start = time.time()
count = 0
try:
    while True:
        if n_cams == 1:
            frame, timestep = c.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("lol", frame)
            cv2.waitKey(1)
            count += 1 
        else:
            frames, timesteps = c.read()
            for i, frame in enumerate(frames):
                frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            cv2.namedWindow("Camera 1")
            cv2.namedWindow("Camera 2")
            cv2.namedWindow("Camera 3")
            cv2.namedWindow("Camera 4")

            cv2.moveWindow("Camera 1", 650, 520)
            cv2.moveWindow("Camera 2", 0, 520)
            cv2.moveWindow("Camera 3", 0, 0)
            cv2.moveWindow("Camera 4", 650, 0)

            cv2.imshow("Camera 1", frames[0])
            cv2.imshow("Camera 2", frames[1])
            cv2.imshow("Camera 3", frames[2])
            cv2.imshow("Camera 4", frames[3])

            cv2.waitKey(1)
            # for i, frame in enumerate(frames):
            #     frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            #     # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            #     cv2.imshow(f"Cam {i+1}", frame)
            # cv2.waitKey(1)
            count += 1
except:
    seconds_elapsed = time.time() - start
    print(f"frames recorded in {seconds_elapsed} sec: {count}| frames per second: {count  / seconds_elapsed}")

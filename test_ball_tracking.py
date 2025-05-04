from pseyepy import Camera
from mocap import Mocap
import time
import cv2
import numpy as np

# BLACK HSV BOUNDS
LOWER = np.array([0., 0., 0.])
UPPER = np.array([50., 50., 50.])

def main():
    mocap = Mocap(fps=150)
    start = time.time()
    frames = 0
    track_ball_time = 0
    try:
        while True:
            iter_start = time.time()
            img, center = mocap.track_ball(lower=LOWER,
                                              upper=UPPER)
            iter_end = time.time()
            track_ball_time += iter_end - iter_start

            dot = cv2.circle(img, center, radius=3, color=[0, 0, 255])
            cv2.imshow("ball tracking", dot)
            cv2.waitKey(1)
            frames += 1          
    except BaseException as e:
        print(e)
        seconds_elapsed = time.time() - start
        fps = frames/seconds_elapsed
        print(f"{frames} recorded in {seconds_elapsed} sec -> fps: {fps}")
        print(f"avg time for mocap.track_ball -> {track_ball_time/frames}")


if __name__ == "__main__":
    main()
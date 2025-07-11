from pseyepy import Camera
from src.mocap import PsEyeMocap
import time
import cv2
import numpy as np
import argparse
import yaml
import src.utils as utils

# BLACK HSV BOUNDS
LOWER = np.array([50, 50, 50], dtype=np.uint8)
UPPER = np.array([255, 255, 255], dtype=np.uint8)

def main(args):
    with open(args.mocap_cfg, "r") as file:
        cfg = yaml.safe_load(file)
    mocap = PsEyeMocap(cfg)
    breakpoint()
    start = time.time()
    frames = 0
    track_ball_time = 0
    try:
        while True:
            iter_start = time.time()
            # img = mocap.read_cameras()[np.newaxis, :, :]
            imgs = mocap.read_cameras()
            imgs = imgs.copy()
            centers = mocap.locate_centers(imgs=imgs,
                                          num_centers=1,
                                          lower=LOWER,
                                          upper=UPPER)
            centers = centers.reshape((centers.shape[0] * centers.shape[1], centers.shape[2]))
            print(centers, centers.shape)
            iter_end = time.time()
            track_ball_time += iter_end - iter_start
            pt_3d = utils.DLT(centers, mocap.projections_wf)
            print(pt_3d)
            mocap.render()
            # frames += 1
            for i in range(len(imgs)):
                img = imgs[i]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                center = centers[i]
                dot = cv2.circle(img, (int(center[0]), int(center[1])), radius=3, color=[0, 0, 255])
                cv2.imshow(f"Cam {i+1}", dot)
                cv2.waitKey(1)
            frames += 1          
    except BaseException as e:
        print(e)
        seconds_elapsed = time.time() - start
        fps = frames/seconds_elapsed
        print(f"{frames} recorded in {seconds_elapsed} sec -> fps: {fps}")
        print(f"avg time for mocap.track_ball -> {track_ball_time/frames}")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--mocap_cfg', type=str, default='cfgs/PSEyeMocap.yaml')
    args = parser.parse_args()  
    main(args)
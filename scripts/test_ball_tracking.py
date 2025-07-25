from pseyepy import Camera
from src.mocap import PsEyeMocap
import time
import cv2
import numpy as np
import argparse
import yaml
import src.utils as utils

LOWER = np.array([50, 50, 50], dtype=np.uint8)
UPPER = np.array([255, 255, 255], dtype=np.uint8)

def main(args):
    with open(args.mocap_cfg, "r") as file:
        cfg = yaml.safe_load(file)
    mocap = PsEyeMocap(cfg)
    start = time.time()
    frames = 0
    all_pts = []
    prev_pt = np.zeros((2, 3), dtype=np.float32)
    try:
        while True:
            iter_start = time.time()
            imgs = mocap.read_cameras()
            imgs = imgs.copy()
            n_centers = 2
            centers, correspondences = mocap.locate_centers(imgs=imgs,
                                          num_centers=n_centers,
                                          lower=LOWER,
                                          upper=UPPER)
            print("centers:")
            print(centers)
            centers = centers.transpose((1, 0, 2)).reshape((centers.shape[0] * centers.shape[1], centers.shape[2]))
            print("correspondences:")
            print(correspondences)
            print("------------------")
            
            # centers = mocap.undistort_points(centers) # TODO THIS ONLY WORKS WITH NUM_CENTERS=1 IN LOCATE_CENTERS RIGHT NOW!!!!!!!!
            pts_3d = np.empty((n_centers, 3), dtype=np.float32)
            for i in range(n_centers):
                pt_3d = utils.DLT(correspondences[i], mocap.projections_wf)
                pts_3d[i] = pt_3d
            # if frames % 2 == 0:
            #     all_pts.append(pt_3d)
                # print(all_pts)
            # print(centers)
            # print(pt_3d)
            # print("-------------------------")
            
            mocap.render(centers=centers,
                         imgs=imgs,
                         pts_3d=pts_3d)
            # for i in range(len(imgs)):
            #     img = imgs[i]
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #     center = centers[i]
            #     dot = cv2.circle(img, (int(center[0]), int(center[1])), radius=3, color=[0, 0, 255])
            #     cv2.imshow(f"Cam {i+1}", dot)
            #     cv2.waitKey(1)
            frames += 1  
            print(pts_3d)
            print(f"dist btwn points: {np.linalg.norm(pts_3d[0] - pts_3d[1])}")
            print(f"diff to prev: {np.linalg.norm(pts_3d - prev_pt)}")
            # if np.linalg.norm(pts_3d - prev_pt) > 0.02:
            #     print("BIG DIFFERENCE\n\n\n\n")
            # if np.linalg.norm(pts_3d - prev_pt) > .05:
            #     print("BIG DIFFERENCE")
            #     breakpoint()
            prev_pt = pts_3d
    except BaseException as e:
        print(e)
        seconds_elapsed = time.time() - start
        fps = frames/seconds_elapsed
        print(f"{frames} recorded in {seconds_elapsed} sec -> fps: {fps}")
        # print(f"avg time for mocap.track_ball -> {track_ball_time/frames}")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--mocap_cfg', type=str, default='cfgs/PSEyeMocap.yaml')
    args = parser.parse_args()  
    main(args)
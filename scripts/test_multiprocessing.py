from src.mocap import PsEyeMocap
import yaml
import numpy as np
import src.utils as utils
import time

LOWER = np.array([50, 50, 50], dtype=np.uint8)
UPPER = np.array([255, 255, 255], dtype=np.uint8)

with open("cfgs/PSEyeMocap.yaml", "r") as file:
    cfg = yaml.safe_load(file)

mocap = PsEyeMocap(cfg)

mocap_frames = 0
start = time.time()
try:
    while True:
        imgs = mocap.read_cameras()
        imgs = imgs.copy()
        centers = mocap.locate_centers(imgs=imgs,
                                        num_centers=1,
                                        lower=LOWER,
                                        upper=UPPER)
        centers = centers.reshape((centers.shape[0] * centers.shape[1], centers.shape[2]))
        centers = mocap.undistort_points(centers) # TODO THIS ONLY WORKS WITH NUM_CENTERS=1 IN LOCATE_CENTERS RIGHT NOW!!!!!!!!
        pt_3d = utils.DLT(centers, mocap.projections_wf)[np.newaxis, :]
        print(centers)
        print(pt_3d)
        print("-------------------------")
        mocap.render(centers=centers,
                        imgs=imgs,
                        pts_3d=pt_3d)
        mocap_frames += 1
except BaseException as e:
    print(e)
    mocap.stop_event.set()
    elapsed = time.time() - start
    print(f"recorded {mocap_frames} in {elapsed} sec -> fps: {mocap_frames/elapsed}")

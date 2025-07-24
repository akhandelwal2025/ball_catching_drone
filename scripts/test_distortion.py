import numpy as np
import cv2
from pseyepy import Camera
import yaml
from src.mocap import PsEyeMocap
# c = Camera(fps=150, 
#            resolution=Camera.RES_SMALL,
#            gain=48,
#            exposure=128)
# pts_2d = np.load('data/pts_2d.npz')['pts_2d']
LOWER = np.array([50, 50, 50], dtype=np.uint8)
UPPER = np.array([255, 255, 255], dtype=np.uint8)
with open("cfgs/PSEyeMocap.yaml", "r") as file:
        cfg = yaml.safe_load(file)
mocap = PsEyeMocap(cfg)
cam1_arrs = np.load('cam1_redone.npz')
K = cam1_arrs['intrinsics']
dist = cam1_arrs['distortion_coeffs']
while True:
    imgs = mocap.read_cameras()
    imgs = imgs.copy()
    centers, correspondences = mocap.locate_centers(imgs=imgs[np.newaxis, :],
                                    num_centers=1,
                                    lower=LOWER,
                                    upper=UPPER)
    centers_undistorted = cv2.undistortPoints(centers, K, dist, P=K)
    mocap.render(centers, imgs=imgs, pts_3d=None)
    print(f"before: {centers}")
    print(f"after: {centers_undistorted}")
    print("-----------------")

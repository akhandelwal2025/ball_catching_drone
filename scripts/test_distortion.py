import numpy as np
import cv2
from pseyepy import Camera

# c = Camera(fps=150, 
#            resolution=Camera.RES_SMALL,
#            gain=63,
#            exposure=255)
pts_2d = np.load('data/pts_2d.npz')['pts_2d']
cam1_arrs = np.load('cam1.npz')
K = cam1_arrs['intrinsics']
dist = cam1_arrs['distortion_coeffs']
n_obs = pts_2d.shape[0] // 4
for i in range(n_obs):
    pt1 = pts_2d[4*i].reshape((1, 1, 2))
    undistorted = cv2.undistortPoints(pt1, K, dist, P=K)
    print(f"og_pt: {pt1} ")
    print(f"undistorted: {undistorted}")
    print("----------------------")

# while True:
#     frame, timestep = c.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     undistorted = cv2.undistort(frame, K, dist)
#     cv2.imshow("original", frame)
#     cv2.imshow("undistorted", undistorted)
#     cv2.waitKey(1)

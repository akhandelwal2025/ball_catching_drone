import numpy as np
import cv2
from pseyepy import Camera

c = Camera(fps=150, 
           resolution=Camera.RES_SMALL,
           gain=63,
           exposure=255)
cam1_arrs = np.load('cam1.npz')
K = cam1_arrs['intrinsics']
dist = cam1_arrs['distortion_coeffs']
while True:
    frame, timestep = c.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    undistorted = cv2.undistort(frame, K, dist)
    cv2.imshow("original", frame)
    cv2.imshow("undistorted", undistorted)
    cv2.waitKey(1)

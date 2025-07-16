from src.mocap import PsEyeMocap
import yaml
import numpy as np
import cv2

with open('cfgs/PSEyeMocap.yaml', "r") as file:
        cfg = yaml.safe_load(file)
mocap = PsEyeMocap(cfg)

ext_c1c2 = mocap.extrinsics_c1c[1]
R = ext_c1c2[:3, :3]
t = ext_c1c2[:3, 3]
K1_inv = np.linalg.inv(mocap.intrinsics[0])
K2_inv = np.linalg.inv(mocap.intrinsics[1])

t_x = np.array([
    [0., -t[2], t[1]],
    [t[2], 0., -t[0]],
    [-t[1], t[0], 0.]
])
E = t_x @ R
F12 = K2_inv.T @ E @ K1_inv
breakpoint()

LOWER = np.array([50, 50, 50], dtype=np.uint8)
UPPER = np.array([255, 255, 255], dtype=np.uint8)

while True:
    imgs = mocap.read_cameras()
    imgs = imgs.copy()
    centers = mocap.locate_centers(imgs=imgs,
                                    num_centers=1,
                                    lower=LOWER,
                                    upper=UPPER)
    centers = centers.reshape((centers.shape[0] * centers.shape[1], centers.shape[2]))
    centers = mocap.undistort_points(centers) # TODO THIS ONLY WORKS WITH NUM_CENTERS=1 IN LOCATE_CENTERS RIGHT NOW!!!!!!!!
    print(centers)
    c1_pt = centers[0]
    c2_epiline = cv2.computeCorrespondEpilines(c1_pt.reshape(1, 1, 2), 1, F12)
    c2_epiline = c2_epiline.reshape(-1, 3)
    print(c2_epiline)

    # vis cam 1
    img1 = imgs[0]
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = cv2.circle(img1, (int(c1_pt[0]), int(c1_pt[1])), radius=3, color=[0, 0, 255])
    cv2.imshow(f"Cam 1", img1)

    # vis cam 2
    img2 = imgs[1]
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    c2_pt = centers[1]
    img2 = cv2.circle(img2, (int(c2_pt[0]), int(c2_pt[1])), radius=3, color=[0, 0, 255])

    # vis epiline
    a, b, c = c2_epiline[0]
    x0, x1 = 0, img2.shape[1]
    y0 = int(round(-(a * x0 + c) / b))
    y1 = int(round(-(a * x1 + c) / b))
    print(x0, y0)
    print(x1, y1)
    img2 = cv2.line(img2, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=2)

    cv2.imshow(f"Cam 2", img2)
    cv2.waitKey(1)
    print("----------------------")
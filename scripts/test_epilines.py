from src.mocap import PsEyeMocap
import yaml
import numpy as np
import cv2

def Fs_through_mocap(mocap):
    Fs = []
    for i in range(1, 4):
        ext_c1ci = mocap.extrinsics_c1c[i]
        R = ext_c1ci[:3, :3]
        t = ext_c1ci[:3, 3]
        K1_inv = np.linalg.inv(mocap.intrinsics[0])
        Ki_inv = np.linalg.inv(mocap.intrinsics[i])

        t_x = np.array([
            [0., -t[2], t[1]],
            [t[2], 0., -t[0]],
            [-t[1], t[0], 0.]
        ])
        E = t_x @ R
        F1i = Ki_inv.T @ E @ K1_inv
        Fs.append(F1i)
    return tuple(Fs)

def Fs_through_points(pts_2d):
    Fs = []
    pts_1 = pts_2d[0]
    for i in range(1, 4):
        pts_i = pts_2d[i]
        F1i, _ = cv2.findFundamentalMat(pts_1, pts_i, method=cv2.FM_RANSAC)
        Fs.append(F1i)
    return tuple(Fs)

with open('cfgs/PSEyeMocap.yaml', "r") as file:
        cfg = yaml.safe_load(file)
mocap = PsEyeMocap(cfg)
mocap_Fs = Fs_through_mocap(mocap)

all_pts = np.load("data/pts_2d.npz")['pts_2d']
pts_2d = np.empty((4, 50, 2), dtype=np.float32)
pts_2d[0] = all_pts[0::4]
pts_2d[1] = all_pts[1::4]
pts_2d[2] = all_pts[2::4]
pts_2d[3] = all_pts[3::4]
pts_Fs = Fs_through_points(pts_2d)

F12, F13, F14 = pts_Fs

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
    # vis cam 1
    img1 = imgs[0]
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = cv2.circle(img1, (int(c1_pt[0]), int(c1_pt[1])), radius=3, color=[0, 0, 255])
    cv2.imshow(f"Cam 1", img1)

    for i, F in [(1, F12), (2, F13), (3, F14)]:
        c2_epiline = cv2.computeCorrespondEpilines(c1_pt.reshape(1, 1, 2), 1, F)
        c2_epiline = c2_epiline.reshape(-1, 3)
        print(c2_epiline)


        # vis cam 2
        img2 = imgs[i]
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        c2_pt = centers[i]
        img2 = cv2.circle(img2, (int(c2_pt[0]), int(c2_pt[1])), radius=3, color=[0, 0, 255])

        # vis epiline
        a, b, c = c2_epiline[0]
        x0, x1 = 0, img2.shape[1]
        y0 = int(round(-(a * x0 + c) / b))
        y1 = int(round(-(a * x1 + c) / b))
        print(x0, y0)
        print(x1, y1)
        img2 = cv2.line(img2, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=1)

        cv2.imshow(f"Cam {i+1}", img2)
        print("----------------------")
    cv2.waitKey(1)
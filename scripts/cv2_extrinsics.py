import cv2
import numpy as np
import yaml
import src.utils as utils
from src.mocap import PsEyeMocap

pts_2d = np.load('data/pts_2d.npz')['pts_2d']
pts1 = pts_2d[0::4, :]
pts2 = pts_2d[1::4, :]
pts3 = pts_2d[2::4, :]
pts4 = pts_2d[3::4, :]

with open('cfgs/PSEyeMocap.yaml', "r") as file:
    cfg = yaml.safe_load(file)

intrinsics = np.empty((4, 3, 3))
for i, cam in enumerate(['cam1', 'cam2', 'cam3', 'cam4']):
    fx = cfg['intrinsics'][cam]['fx'] 
    fy = cfg['intrinsics'][cam]['fy']
    ox = cfg['intrinsics'][cam]['ox'] 
    oy = cfg['intrinsics'][cam]['oy']
    intrinsics[i, :, :] = utils.construct_intrinsics(fx, fy, ox, oy)

# Convert pts1 and pts2 to Nx3 homogeneous coordinates
pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3).T  # shape: 3 x N
pts2_h = cv2.convertPointsToHomogeneous(pts2).reshape(-1, 3).T
pts3_h = cv2.convertPointsToHomogeneous(pts3).reshape(-1, 3).T
pts4_h = cv2.convertPointsToHomogeneous(pts4).reshape(-1, 3).T

# Apply inverse intrinsics to get normalized coordinates
K1_inv = np.linalg.inv(intrinsics[0])
K2_inv = np.linalg.inv(intrinsics[1])
K3_inv = np.linalg.inv(intrinsics[2])
K4_inv = np.linalg.inv(intrinsics[3])

pts1_norm = (K1_inv @ pts1_h).T[:, :2]  # shape: N x 2
pts2_norm = (K2_inv @ pts2_h).T[:, :2]
pts3_norm = (K3_inv @ pts3_h).T[:, :2]
pts4_norm = (K4_inv @ pts4_h).T[:, :2]

# Camera 1 to Camera 2
E12, _ = cv2.findEssentialMat(pts1_norm, pts2_norm, method=cv2.RANSAC, prob=0.999, threshold=1.0)
_, R12, t12, _ = cv2.recoverPose(E12, pts1_norm, pts2_norm)
t12_unit = t12 / np.linalg.norm(t12)
t12_scaled = t12_unit * 3.368

# Camera 1 to Camera 3
E13, _ = cv2.findEssentialMat(pts1_norm, pts3_norm, method=cv2.RANSAC, prob=0.999, threshold=1.0)
_, R13, t13, _ = cv2.recoverPose(E13, pts1_norm, pts3_norm)
t13_unit = t13 / np.linalg.norm(t13)
t13_scaled = t13_unit * 4.779

# Camera 1 to Camera 4
E14, _ = cv2.findEssentialMat(pts1_norm, pts4_norm, method=cv2.RANSAC, prob=0.999, threshold=1.0)
_, R14, t14, _ = cv2.recoverPose(E14, pts1_norm, pts4_norm)
t14_unit = t14 / np.linalg.norm(t14)
t14_scaled = t14_unit * 3.39

ext_c1c = np.empty((4, 3, 4))
ext_c1c[0] = np.hstack((np.eye(3), np.zeros((3, 1))))
ext_c1c[1] = np.hstack((R12, t12))
ext_c1c[2] = np.hstack((R13, t13))
ext_c1c[3] = np.hstack((R14, t14))

Ps = np.empty((4, 3, 4))
Ps[0] = intrinsics[0] @ ext_c1c[0]
Ps[1] = intrinsics[1] @ ext_c1c[1]
Ps[2] = intrinsics[2] @ ext_c1c[2]
Ps[3] = intrinsics[3] @ ext_c1c[3]

mocap = PsEyeMocap(cfg)
ext_c1w = mocap.extrinsics_cw[0]
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
    pt_3d = utils.DLT(pixels=centers, projections=Ps)[np.newaxis, :]
    # pt_3d = utils.transform(ext_c1w, pt_3d)
    print(pt_3d)
    for i in range(len(imgs)):
        img = imgs[i]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        center = centers[i]
        dot = cv2.circle(img, (int(center[0]), int(center[1])), radius=3, color=[0, 0, 255])
        cv2.imshow(f"Cam {i+1}", dot)
        cv2.waitKey(1)
breakpoint()
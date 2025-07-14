from pseyepy import Camera
from src.mocap import PsEyeMocap
import time
import cv2
import numpy as np
import argparse
import yaml
import src.utils as utils
from scipy.spatial.transform import Rotation as R 
from scipy.optimize import least_squares

# BLACK HSV BOUNDS
LOWER = np.array([50, 50, 50], dtype=np.uint8)
UPPER = np.array([255, 255, 255], dtype=np.uint8)
N_FRAMES = 20
def main(args):
    with open(args.mocap_cfg, "r") as file:
        cfg = yaml.safe_load(file)
    mocap = PsEyeMocap(cfg)
    # i = 0
    # pts_2d = np.empty((N_FRAMES, 4, 2))
    # while i < N_FRAMES:
    #     imgs = mocap.read_cameras()
    #     imgs = imgs.copy()
    #     centers = mocap.locate_centers(imgs=imgs,
    #                                     num_centers=1,
    #                                     lower=LOWER,
    #                                     upper=UPPER)
    #     centers = centers.reshape((centers.shape[0] * centers.shape[1], centers.shape[2]))
    #     print(centers)
    #     print(f"saved {i} of {N_FRAMES}")
    #     print("-----------------")
    #     for j in range(len(imgs)):
    #         img = imgs[j]
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         center = centers[j]
    #         dot = cv2.circle(img, (int(center[0]), int(center[1])), radius=3, color=[0, 0, 255])
    #         cv2.imshow(f"Cam {j+1}", dot)
    #     key = cv2.waitKey(1) & 0xFF
    #     if not np.any(centers == 0.) and key != 255:
    #         pts_2d[i] = centers
    #         i += 1
    # pts_2d = pts_2d.reshape(N_FRAMES * 4, 2)
    # np.savez("data/pts_2d.npz", pts_2d = pts_2d)
    # breakpoint()
    pts_2d = np.load("data/pts_2d.npz")['pts_2d']
    # print("in main loop:")
    # print(mocap.projections_wf)
    x0 = np.empty((4*2, 3), dtype=np.float32)
    x0[0::2] = mocap.extrinsics_wc[:, :3, 3].reshape((4, 3))
    x0[1::2] = R.from_matrix(mocap.extrinsics_wc[:, :3, :3]).as_rotvec()
    # residuals = utils.ba_calc_residuals(x0.flatten(), 4, mocap.intrinsics, centers)
    res = least_squares(fun=utils.ba_calc_residuals,
                        x0=x0.flatten(),
                        loss='huber',
                        # f_scale=4.0,
                        # jac='3-point',
                        # x_scale='jac',
                        ftol=2.2e-16,
                        xtol=2.2e-16,
                        verbose=2,
                        args=(4, mocap.intrinsics, pts_2d))
    print(res.x)
    print(res.fun)
    breakpoint()
    cam_pos = np.zeros((4, 3))
    cam_eulers = np.zeros((4, 3))
    output = res.x.reshape((8, 3))
    for i in range(4):
        pos = output[2*i, :]
        rot_vec = output[2*i + 1, :]
        rot_mtrx = R.from_rotvec(rot_vec).as_matrix()
        eulers = R.from_rotvec(rot_vec).as_euler("ZYX", degrees=True)
        assert np.allclose(R.from_euler('ZYX', np.radians(eulers)).as_matrix(), rot_mtrx) 
        pos = -rot_mtrx.T @ pos
        cam_pos[i] = pos
        cam_eulers[i] = eulers
    print(cam_pos)
    print(cam_eulers)
    # print(residuals)
    # print(f'mean: {residuals.mean()}')
    # print(f'norm: {np.linalg.norm(residuals) ** 2}')
    # print("--------------------------")
    # pt_3d = utils.DLT(centers, mocap.projections_wf)
    
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--mocap_cfg', type=str, default='cfgs/PSEyeMocap.yaml')
    args = parser.parse_args()  
    main(args)
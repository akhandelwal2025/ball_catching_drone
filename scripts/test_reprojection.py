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
N_FRAMES = 30

def collect_imgs(mocap, n_frames):
    i = 0
    pts_2d = np.empty((n_frames, 4, 2))
    while i < n_frames:
        imgs = mocap.read_cameras()
        imgs = imgs.copy()
        centers = mocap.locate_centers(imgs=imgs,
                                        num_centers=1,
                                        lower=LOWER,
                                        upper=UPPER)
        centers = centers.reshape((centers.shape[0] * centers.shape[1], centers.shape[2]))
        print(centers)
        print(f"saved {i} of {n_frames}")
        print("-----------------")
        for j in range(len(imgs)):
            img = imgs[j]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            center = centers[j]
            dot = cv2.circle(img, (int(center[0]), int(center[1])), radius=3, color=[0, 0, 255])
            cv2.imshow(f"Cam {j+1}", dot)
        key = cv2.waitKey(1) & 0xFF
        if not np.any(centers == 0.) and key != 255:
            pts_2d[i] = centers
            i += 1
    pts_2d = pts_2d.reshape(n_frames * 4, 2)
    np.savez("data/pts_2d.npz", pts_2d = pts_2d)
    return pts_2d

def undistort_points(pts_2d):
    cam1_arrs = np.load("data/zoomed_intrinsics/cam1.npz")
    cam2_arrs = np.load("data/zoomed_intrinsics/cam2.npz")
    cam3_arrs = np.load("data/zoomed_intrinsics/cam3.npz")
    cam4_arrs = np.load("data/zoomed_intrinsics/cam4.npz")

    cam1_K = cam1_arrs['intrinsics']
    cam1_dist = cam1_arrs['distortion_coeffs']

    cam2_K = cam2_arrs['intrinsics']
    cam2_dist = cam2_arrs['distortion_coeffs']

    cam3_K = cam3_arrs['intrinsics']
    cam3_dist = cam3_arrs['distortion_coeffs']

    cam4_K = cam4_arrs['intrinsics']
    cam4_dist = cam4_arrs['distortion_coeffs']

    n_obs = pts_2d.shape[0] // 4
    pts_2d_undistorted = np.empty(pts_2d.shape, dtype=np.float32)
    for i in range(n_obs):
        pt1 = pts_2d[4*i]
        pt2 = pts_2d[4*i+1]
        pt3 = pts_2d[4*i+2]
        pt4 = pts_2d[4*i+3]
        pts_2d_undistorted[4*i] = cv2.undistortPoints(pt1, cam1_K, cam1_dist, P=cam1_K)
        pts_2d_undistorted[4*i+1] = cv2.undistortPoints(pt2, cam2_K, cam2_dist, P=cam2_K)
        pts_2d_undistorted[4*i+2] = cv2.undistortPoints(pt3, cam3_K, cam3_dist, P=cam3_K)
        pts_2d_undistorted[4*i+3] = cv2.undistortPoints(pt4, cam4_K, cam4_dist, P=cam4_K)
    return pts_2d_undistorted

def main(args):
    with open(args.mocap_cfg, "r") as file:
        cfg = yaml.safe_load(file)
    mocap = PsEyeMocap(cfg)
    if args.imgs == '':
        pts_2d = collect_imgs(mocap, args.n_frames)
    else:
        pts_2d = np.load("data/pts_2d.npz")['pts_2d']
    
    pts_2d = undistort_points(pts_2d)
    x0 = np.empty((34), dtype=np.float32)
    for i in range(4):
        for i, cam in enumerate(['cam1', 'cam2', 'cam3', 'cam4']):
            fx = cfg['intrinsics'][cam]['fx'] 
            fy = cfg['intrinsics'][cam]['fy']
            ox = cfg['intrinsics'][cam]['ox'] 
            oy = cfg['intrinsics'][cam]['oy']
            pos = mocap.extrinsics_c1c[i, :3, 3].reshape(1, 3)
            rotvec = R.from_matrix(mocap.extrinsics_c1c[i, :3, :3]).as_rotvec()
            if i == 0:
                x0[0] = fx
                x0[1] = fy
                x0[2] = ox
                x0[3] = oy
            else:
                x0[4+10*(i-1):4+10*(i-1)+3] = pos
                x0[4+10*(i-1)+3:4+10*(i-1)+6] = rotvec
                x0[4+10*(i-1)+6] = fx
                x0[4+10*(i-1)+7] = fy
                x0[4+10*(i-1)+8] = ox
                x0[4+10*(i-1)+9] = oy
    print("in main")
    print(mocap.projections_c1f)
    og_residuals = utils.ba_calc_residuals(x0, 4, pts_2d)
    breakpoint()
    res = least_squares(fun=utils.ba_calc_residuals,
                        x0=x0,
                        loss='huber',
                        # f_scale=4.0,
                        # jac='3-point',
                        # x_scale='jac',
                        ftol=2.2e-16,
                        xtol=2.2e-16,
                        verbose=2,
                        args=(4, pts_2d))
    print(res.x)
    print(res.fun)
    output = res.x
    intrinsics =  np.empty((4, 3, 3), dtype=np.float32)
    ext_c1cs = np.empty((4, 3, 4), dtype=np.float32)
    ext_wcs = np.empty((4, 3, 4), dtype=np.float32)
    Ps = np.empty((4, 3, 4), dtype=np.float32)
    ext_wc1 = mocap.extrinsics_wc[0]
    for i in range(4):
        if i == 0:
            fx = output[0]
            fy = output[1]
            ox = output[2]
            oy = output[3]
            intrinsics[i] = utils.construct_intrinsics(fx, fy, ox, oy)
            ext_c1cs[i] = np.hstack((np.eye(3), np.zeros((3, 1))))
            ext_wcs[i] = utils.compose_Ps(ext_c1cs[i], ext_wc1)
            Ps[i] = intrinsics[i] @ ext_wcs[i]
            print("cam1:")
            print(f"    fx: {fx}")
            print(f"    fy: {fy}")
            print(f"    ox: {ox}")
            print(f"    oy: {oy}")
        else:
            params = res.x[4 + 10*(i-1):4 + 10*i]
            pos = params[:3].reshape((3, 1))
            rot_vec = params[3:6].reshape((3,))
            fx, fy, ox, oy = params[6], params[7], params[8], params[9]
            intrinsics[i] = utils.construct_intrinsics(fx, fy, ox, oy)

            rot_mtrx = R.from_rotvec(rot_vec).as_matrix()
            ext_c1cs[i] = np.hstack((rot_mtrx, pos))
            ext_wcs[i] = utils.compose_Ps(ext_c1cs[i], ext_wc1)
            Ps[i] = intrinsics[i] @ ext_wcs[i]
            R_wf = ext_wcs[i, :3, :3]
            transform = np.array([
                [0., 1., 0.,], 
                [0., 0., 1.], 
                [1., 0., 0.,]
            ])
            eulers = R.from_matrix((transform.T @ R_wf).T).as_euler("ZYX", degrees=True)
            pos = -R_wf.T @ ext_wcs[i, :3, 3]
            print(f"cam{i}:")
            print(f"    fx: {fx}")
            print(f"    fy: {fy}")
            print(f"    ox: {ox}")
            print(f"    oy: {oy}")
            print(f"    pos: {np.round(pos, 3)}")
            print(f"    eulers: {np.round(eulers[::-1], 3)}")
    
    # evaluate reprojection error
    n_cams = 4
    n_obs = pts_2d.shape[0] // n_cams
    obs_3d = np.empty((n_obs, 3), dtype=np.float32)
    for i in range(n_obs):
        obs_3d[i] = utils.DLT(pixels=pts_2d[n_cams*i:n_cams*(i+1)],
                        projections=Ps)

    # calculate residuals
    obs_3d = obs_3d.T                                                       # (3, n_obs)
    obs_3d = np.vstack((obs_3d, np.ones((1, obs_3d.shape[1]))))             # add homo coords, (4, n_obs)
    projected_2d = Ps @ obs_3d                                              # (n_cams, 3, 4) @ (4, n_obs) = (n_cams, 3, n_obs)
    projected_2d = np.transpose(projected_2d, (2, 0, 1)).reshape(-1, 3)     # (n_obs * n_cams, 3). ordering of (2, 0, 1) is important to preserve proper interleaving
    projected_2d = projected_2d / projected_2d[:, -1][:, np.newaxis]
    projected_2d = projected_2d[:, :2]                                      # get rid of homo coords, (n_obs * n_cams, 2)
    new_residuals = (projected_2d-pts_2d).flatten()
    print(f"np.mean(og_residuals -> new_residuals): {np.mean(og_residuals)} -> {np.mean(new_residuals)}")
    print(f"np.linalg.norm(og_residuals -> new_residuals): {np.linalg.norm(og_residuals)} -> {np.linalg.norm(new_residuals)}")
    breakpoint()




    # ext_c1c = np.empty((4, 3, 4))
    # ext_c1c[0] = np.hstack((np.eye(3), np.zeros((3, 1))))
    # for i in range(3):
    #     params = output[6*i:6*(i+1)]
    #     pos = params[:3].reshape((3, 1))
    #     rot_vec = params[3:6].reshape((3,))
    #     rot_mtrx = R.from_rotvec(rot_vec).as_matrix()
    #     ext_c1c[i+1] = np.hstack((rot_mtrx, pos))
    
    # ext_wc1 = mocap.extrinsics_wc[0]
    # ext_wcs = np.empty((4, 3, 4))
    # Ps = np.empty((4, 3, 4))
    # cam_pos = np.zeros((4, 3))
    # cam_eulers = np.zeros((4, 3))
    # for i in range(4):
    #     ext_wc = utils.compose_Ps(ext_c1c[i], ext_wc1)
    #     ext_wcs[i] = ext_wc
    #     Ps[i] = mocap.intrinsics[i] @ ext_wc
    #     rot_mtrx = ext_wc[:3, :3]
    #     eulers = R.from_matrix(rot_mtrx).as_euler("ZYX", degrees=True)
    #     pos = -rot_mtrx.T @ ext_wc[:3, 3]
    #     cam_pos[i] = pos
    #     cam_eulers[i] = [eulers[2], eulers[1], eulers[0]]

    #     z, y, x = eulers[0], eulers[1], eulers[2]
    #     assert np.allclose(R.from_euler("ZYX", [z, y, x], degrees=True).as_matrix(), rot_mtrx)
    #     print("verified rot mtrx")
    # print(ext_wcs)
    # print(cam_pos)
    # print(cam_eulers)

    # evalute reprojection error
    # n_cams = 4
    # n_obs = pts_2d.shape[0] // n_cams
    # obs_3d = np.empty((n_obs, 3), dtype=np.float32)
    # for i in range(n_obs):
    #     obs_3d[i] = utils.DLT(pixels=pts_2d[n_cams*i:n_cams*(i+1)],
    #                     projections=Ps)
    # # calculate residuals
    # obs_3d = obs_3d.T                                                       # (3, n_obs)
    # obs_3d = np.vstack((obs_3d, np.ones((1, obs_3d.shape[1]))))             # add homo coords, (4, n_obs)
    # projected_2d = Ps @ obs_3d                                              # (n_cams, 3, 4) @ (4, n_obs) = (n_cams, 3, n_obs)
    # # breakpoint()
    # projected_2d = np.transpose(projected_2d, (2, 0, 1)).reshape(-1, 3)     # (n_obs * n_cams, 3). ordering of (2, 0, 1) is important to preserve proper interleaving
    # projected_2d = projected_2d / projected_2d[:, -1][:, np.newaxis]
    # projected_2d = projected_2d[:, :2]                                      # get rid of homo coords, (n_obs * n_cams, 2)
    # residuals = (projected_2d-pts_2d).flatten()
    # print(residuals)
    # print(Ps)

    # breakpoint()
    # print(residuals)
    # print(f'mean: {residuals.mean()}')
    # print(f'norm: {np.linalg.norm(residuals) ** 2}')
    # print("--------------------------")
    # pt_3d = utils.DLT(centers, mocap.projections_wf)
    
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--mocap_cfg', type=str, default='cfgs/PSEyeMocap.yaml')
    parser.add_argument('--n_frames', type=int, default=30)
    parser.add_argument('--imgs', type=str, default='data/pts_2d.npz')
    args = parser.parse_args()  
    main(args)
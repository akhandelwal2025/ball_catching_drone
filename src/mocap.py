from pseyepy import Camera, cam_count
from abc import ABC, abstractmethod
import cv2
import numpy as np
import time
import src.utils as utils
from src.vis import Vis
from src.env import TestEnv
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

class FakeCameras():
    def __init__(self,
                 fake_imgs_filepath):
        self.filepath = fake_imgs_filepath
        self.i = 0
        self.imgs = np.load(fake_imgs_filepath)['fake_imgs']
        self.N = self.imgs.shape[0]

    def read(self):
        # mirror pseyepy .read() function which returns imgs, timestamps
        imgs = self.imgs[self.i]
        timestep = time.time()
        self.i = (self.i + 1) % self.N
        return imgs, timestep

class BaseMocap(ABC):
    def __init__(self, mocap_cfg):
        self.mocap_cfg = mocap_cfg
        self.vis = Vis(self.mocap_cfg,
                       self.cam_pos,
                       self.cam_eulers)

        self.construct_intrinsics()
        self.construct_extrinsics_wf()
        self.construct_extrinsics_c1f()
        self.construct_projections()
    
    @abstractmethod
    def construct_intrinsics(self):
        pass
    
    @abstractmethod
    def construct_extrinsics_wf(self):
        pass
    
    def construct_extrinsics_c1f(self):
        pass

    @abstractmethod
    def construct_projections(self):
        pass
    
    @abstractmethod
    def read_cameras(self):
        pass
    
    @abstractmethod
    def render(self):
        pass

    def to_cam1(self, extrinsics_wc, extrinsics_cw):
        """
        Given a set of extrinsics defined in a world frame, transform all extrinsics to be relative to cam1
        with this transformation, the first extrinsic matrix that results will be of the form [I|0]
        Inputs:
            extrinsics_wc: np.ndarray - world -> camera extrinsics. shape = (N, 3, 4)
            extrinsics_cw: np.ndarray - camera -> world extrinsics. shape = (N, 3, 4)
        Returns:
            ext_cc1, cam_i to cam_1. shape = (N, 3, 4)
            ext_c1c, cam_1 to cam_i. shape = (N, 3, 4)
            In both cases, first matrix will be of the form [I|0]
        """
        ext_wc1 = np.tile(extrinsics_wc[0], (self.n_cams, 1, 1))
        ext_c1w = np.tile(extrinsics_cw[0], (self.n_cams, 1, 1))
        ext_cc1 = utils.compose_Ps(ext_wc1, extrinsics_cw)
        ext_c1c = utils.compose_Ps(extrinsics_wc, ext_c1w)
        # ext_cc1 = ext_wc1 @ extrinsics_cw
        # ext_c1c = extrinsics_wc @ ext_c1w
        return ext_cc1, ext_c1c

    def collect_imgs(self,
                     num_timesteps,
                     save = False):
        mocap_imgs = np.empty((num_timesteps, self.n_cams, self.resolution))
        for i in range(num_timesteps):
            imgs, _ = self.read_cameras()
            mocap_imgs[i] = imgs
        if save:
            filename = f'mocap_{num_timesteps}_frames'
            np.save(filename, mocap_imgs=mocap_imgs)
        return mocap_imgs
    
    def undistort_points(self, pts_2d):
        cam1_K = self.cam1_arrs['intrinsics']
        cam1_dist = self.cam1_arrs['distortion_coeffs']

        cam2_K = self.cam2_arrs['intrinsics']
        cam2_dist = self.cam2_arrs['distortion_coeffs']

        cam3_K = self.cam3_arrs['intrinsics']
        cam3_dist = self.cam3_arrs['distortion_coeffs']

        cam4_K = self.cam4_arrs['intrinsics']
        cam4_dist = self.cam4_arrs['distortion_coeffs']

        n_obs = pts_2d.shape[0] // 4
        pts_2d_undistorted = np.empty(pts_2d.shape, dtype=np.float32)
        for i in range(n_obs):
            pt1 = pts_2d[4*i]
            pt2 = pts_2d[4*i+1]
            pt3 = pts_2d[4*i+2]
            pt4 = pts_2d[4*i+3]
            pts_2d_undistorted[4*i] = cv2.undistortPoints(pt1, cam1_K, cam1_dist, P=cam1_K) if not np.any(pt1 == -1) else pt1
            pts_2d_undistorted[4*i+1] = cv2.undistortPoints(pt2, cam2_K, cam2_dist, P=cam2_K) if not np.any(pt2 == -1) else pt2
            pts_2d_undistorted[4*i+2] = cv2.undistortPoints(pt3, cam3_K, cam3_dist, P=cam3_K) if not np.any(pt3 == -1) else pt3
            pts_2d_undistorted[4*i+3] = cv2.undistortPoints(pt4, cam4_K, cam4_dist, P=cam4_K) if not np.any(pt4 == -1) else pt4
        return pts_2d_undistorted
    
    def locate_centers(self, imgs, num_centers, lower, upper):
        """
        Locate IR dots in set of images
        Inputs:
            imgs: List[img] - images from each camera in mocap system. shape = (num_cams, resolution)
            num_centers: int - expected number of centers to find in each img
            lower: List[int] - lower bound of values to use to create mask
            upper: List[int] - upper bound of values to use to create mask
        """
        num_imgs = imgs.shape[0]
        centers = np.full((num_imgs, num_centers, 2), -1, dtype=np.float32)
        for i in range(num_imgs):
            img = imgs[i]
            # print("img.shape:", img.shape, "img.dtype:", img.dtype)
            # print("lower:", lower, "shape:", np.array(lower).shape, "dtype:", np.array(lower).dtype)
            # print("upper:", upper, "shape:", np.array(upper).shape, "dtype:", np.array(upper).dtype)\
            mask = cv2.inRange(img, lower, upper)
            if not self.mocap_cfg['use_fake_imgs']:
                # mask = cv2.erode(mask, None, iterations=2)
                mask = cv2.dilate(mask, None, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = list(contours)
            contours.sort(key=cv2.contourArea, reverse=True) # assumption is that dots are the biggest thing in the img
            contours = contours[:min(len(contours), num_centers)]
            img_centers = []
            for c in range(num_centers):
                if c < len(contours):
                    M = cv2.moments(contours[c])
                    if M['m00'] != 0:
                        img_centers.append([(M["m10"] / M["m00"]), (M["m01"] / M["m00"])])
            if len(img_centers) != 0:
                centers[i, :len(img_centers), :] = img_centers
        return centers
    
    def bundle_adjustment(self, n_obs):
        obs_2d = np.empty((self.n_cams * n_obs, 2), dtype=int)
        for i in range(n_obs):
            imgs = self.read_cameras()
            pixels = self.locate_centers(imgs, 
                                         num_centers=1, 
                                         lower=200,
                                         upper=256).squeeze()
            obs_2d[self.n_cams*i:self.n_cams*(i+1)] = pixels
        x0 = np.empty((self.n_cams*2, 3), dtype=np.float32)
        x0[0::2] = self.ext_c1c[:, :3, 3].reshape((self.n_cams, 3))
        x0[1::2] = R.from_matrix(self.ext_c1c[:, :3, :3]).as_rotvec()
        x0 = x0[2:] # get rid of first cam
        res = utils.ba_calc_residuals(x0.flatten(), self.n_cams, self.intrinsics, obs_2d)
        res = least_squares(fun=utils.ba_calc_residuals,
                            x0=x0.flatten(),
                            loss='huber',
                            # f_scale=4.0,
                            # jac='3-point',
                            # x_scale='jac',
                            # ftol=2.2e-16,
                            # xtol=2.2e-16,c
                            verbose=2,
                            args=(self.n_cams, self.intrinsics, obs_2d))
        
        # ---------- EVALUATE BA ----------
        # gt_projections = self.env.gt_projections
        # og_projections = self.env.projections
        # new_projections = np.empty((self.n_cams, 3, 4))
        # ext_wc1 = self.env.gt_extrinsics_wc[0]     
        # for i in range(self.n_cams):
        #     if i == 0:
        #         new_projections[0] = self.intrinsics[0] @ ext_wc1
        #     else:
        #         params = res.x[6*(i-1):6*i]
        #         pos = params[:3].reshape((3, 1))
        #         rot_vec = params[3:6].reshape((3,))
        #         rot_mtrx = R.from_rotvec(rot_vec).as_matrix()
        #         ext_c1c = np.hstack((rot_mtrx, pos))
        #         intrinsic = self.intrinsics[i, :, :]
        #         new_projections[i, :, :] = intrinsic @ utils.compose_Ps(ext_c1c, ext_wc1) 
        # n_eval = 10
        # obs_2d = np.empty((self.n_cams * n_eval, 2), dtype=int)
        # obs_3d = np.empty((n_eval, 3), dtype=np.float32)
        # for i in range(n_eval):
        #     imgs = self.read_cameras()
        #     # pixels = self.env.projected_pts
        #     pixels = self.locate_centers(imgs, 
        #                                     num_centers=1, 
        #                                     lower=200,
        #                                     upper=256).squeeze()
        #     obs_2d[self.n_cams*i:self.n_cams*(i+1)] = pixels
        #     obs_3d[i] = self.env.feature_pt
        # gt_3d = utils.project_2d_to_3d(self.n_cams, gt_projections, obs_2d)
        # gt_reproj = utils.project_3d_to_2d(gt_projections, obs_3d)
        # og_pred = utils.project_2d_to_3d(self.n_cams, og_projections, obs_2d)
        # og_reproj = utils.project_3d_to_2d(og_projections, obs_3d)
        # new_pred = utils.project_2d_to_3d(self.n_cams, new_projections, obs_2d)
        # new_reproj = utils.project_3d_to_2d(new_projections, obs_3d)
        # print(f"L2-norm(gt_pred - new_pred): {np.linalg.norm(gt_3d - new_pred, axis=1)}")
        # print(f"L2-norm(gt_pred vs og_pred): {np.linalg.norm(gt_3d - og_pred, axis=1)}")
        # print("------------- PRE CHANGE -------------")
        # print(f'gt_cam_pos: {self.env.gt_cam_pos} | gt_cam_eulers: {self.env.gt_cam_eulers}')
        # print(f'cam_pos: {self.env.cam_pos} | cam_eulers: {self.env.cam_eulers}')
        # print(f"L2-norm(gt_cam_eulers - cam_eulers): {np.degrees(np.linalg.norm(self.env.gt_cam_eulers - self.env.cam_eulers))}")
        # print(f"L2-norm(gt_cam_pos - cam_pos): {np.linalg.norm(self.env.gt_cam_pos - self.env.cam_pos)}")
        # breakpoint()
        # pos = np.empty((self.n_cams, 3))
        # eulers = np.empty((self.n_cams, 3))
        # for i in range(self.n_cams):
        #     ext = np.linalg.inv(self.intrinsics[i]) @ new_projections[i]
        #     R_wc = ext[:, :3]
        #     t_wc = ext[:, 3].reshape(3, 1)
        #     R_cw = R_wc.T
        #     t_cw = -R_cw @ t_wc
        #     pos[i] = t_cw.flatten()
        #     eulers[i] = R.from_matrix(R_cw).as_euler('xyz', degrees=False)
        # self.env.cam_pos = pos
        # self.env.cam_eulers = eulers
        # print("------------- POST CHANGE -------------")
        # print(f'gt_cam_pos: {self.env.gt_cam_pos} | gt_cam_eulers: {self.env.gt_cam_eulers}')
        # print(f'cam_pos: {self.env.cam_pos} | cam_eulers: {self.env.cam_eulers}')
        # print(f"L2-norm(gt_cam_eulers - cam_eulers): {np.degrees(np.linalg.norm(self.env.gt_cam_eulers - self.env.cam_eulers))}")
        # print(f"L2-norm(gt_cam_pos - cam_pos): {np.linalg.norm(self.env.gt_cam_pos - self.env.cam_pos)}")
        # self.env.render()
        # breakpoint()
        # self.env.cam_pos = new_projections[:, :3, 3].reshape((self.n_cams, 3))
        # self.env.cam_eulers = R.from_matrix(new_projections[:, :3, :3].reshape((self.n_cams, 3, 3))).as_euler('xyz', degrees=False)
        # self.env.render()
        # breakpoint()

class FakeMocap(BaseMocap):
    def __init__(self, mocap_cfg, env_cfg):
        self.env = TestEnv(env_cfg)
        self.n_cams = self.env.n_cams
        self.resolution = self.env.resolution

        super().__init__(mocap_cfg)

    def construct_intrinsics(self):
        self.intrinsics = self.env.intrinsics        
    
    def construct_extrinsics_wf(self):
        self.extrinsics_wc = self.env.extrinsics_wc
        self.extrinsics_cw = self.env.extrinsics_cw

    def construct_extrinsics_c1f(self):
        self.extrinsics_cc1, self.extrinsics_c1c = self.to_cam1(self.extrinsics_wc, self.extrinsics_cw)
    
    def construct_projections(self):
        self.projections_wf = self.intrinsics @ self.extrinsics_wc
        self.projections_c1f = self.intrinsics @ self.extrinsics_c1c
         
    def read_cameras(self):
        feature_pt = self.env.gen_random_pt()
        while not self.env.pt_in_fovs(feature_pt) or not self.env.gen_imgs():
            feature_pt = self.env.gen_random_pt()
        self.env.render()
        self.imgs = self.env.fake_imgs
        return self.imgs

    def render(self):
        self.vis.render(imgs=self.imgs,
                        frames3d={
                            'green': (self.env.gt_cam_pos, self.env.gt_cam_eulers, True),
                            'red': (self.env.cam_pos, self.env.cam_eulers, False)
                        },
                        feature_pts=self.env.feature_pt)

class PsEyeMocap(BaseMocap):
    def __init__(self, mocap_cfg):
        self.n_cams = mocap_cfg['n_cams']
        if mocap_cfg['pseyepy_params']['resolution'] == 'small':
            self.resolution = (320, 240)
            mocap_cfg['pseyepy_params']['resolution'] = Camera.RES_SMALL
        else:
            self.resolution = (640, 480)
            mocap_cfg['pseyepy_params']['resolution'] = Camera.RES_LARGE
        self.c = Camera(**mocap_cfg['pseyepy_params'])

        self.cam_pos = np.stack([mocap_cfg['pos']['cam1'],
                                 mocap_cfg['pos']['cam2'],
                                 mocap_cfg['pos']['cam3'],
                                 mocap_cfg['pos']['cam4']], axis=0)
        self.cam_eulers = np.stack([mocap_cfg['eulers']['cam1'],
                                    mocap_cfg['eulers']['cam2'],
                                    mocap_cfg['eulers']['cam3'],
                                    mocap_cfg['eulers']['cam4']], axis=0)
        self.cam_eulers = np.radians(self.cam_eulers)

        self.cam1_arrs = np.load("data/zoomed_intrinsics/cam1.npz")
        self.cam2_arrs = np.load("data/zoomed_intrinsics/cam2.npz")
        self.cam3_arrs = np.load("data/zoomed_intrinsics/cam3.npz")
        self.cam4_arrs = np.load("data/zoomed_intrinsics/cam4.npz")

        super().__init__(mocap_cfg)

    def construct_intrinsics(self):
        self.intrinsics = np.empty((4, 3, 3))
        for i, cam in enumerate(['cam1', 'cam2', 'cam3', 'cam4']):
            fx = self.mocap_cfg['intrinsics'][cam]['fx'] 
            fy = self.mocap_cfg['intrinsics'][cam]['fy']
            ox = self.mocap_cfg['intrinsics'][cam]['ox'] 
            oy = self.mocap_cfg['intrinsics'][cam]['oy']
            self.intrinsics[i, :, :] = utils.construct_intrinsics(fx, fy, ox, oy)
    
    def construct_extrinsics_wf(self):
        self.extrinsics_wc = np.empty((4, 3, 4))
        self.extrinsics_cw = np.empty((4, 3, 4))
        for i in range(self.n_cams):
            ext_wc, ext_cw = utils.construct_extrinsics(pos=self.cam_pos[i][:, np.newaxis],
                                                        eulers=self.cam_eulers[i])
            self.extrinsics_wc[i, :, :] = ext_wc
            self.extrinsics_cw[i, :, :] = ext_cw
    
    def construct_extrinsics_c1f(self):
        self.extrinsics_cc1, self.extrinsics_c1c = self.to_cam1(self.extrinsics_wc, self.extrinsics_cw)
    
    def construct_projections(self):
        self.projections_wf = self.intrinsics @ self.extrinsics_wc
        self.projections_c1f = self.intrinsics @ self.extrinsics_c1c
    
    def read_cameras(self):
        imgs, timesteps = self.c.read()
        self.imgs = np.array(imgs)
        return self.imgs
    
    def render(self,
               centers,
               imgs,
               pts_3d):
        centers = centers.astype(int)
        self.vis.render(centers,
                        imgs,
                        pts_3d)
from pseyepy import Camera, cam_count
from abc import ABC, abstractmethod
import cv2
import numpy as np
import time
import src.utils as utils
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

        self.construct_intrinsics()
        self.construct_extrinsics()
        # self.construct_projections()
    
    @abstractmethod
    def construct_intrinsics(self):
        pass
    
    @abstractmethod
    def construct_extrinsics(self):
        pass

    @abstractmethod
    def construct_projections(self):
        pass
    
    @abstractmethod
    def read_cameras(self):
        pass

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
        centers = np.full((num_imgs, num_centers, 2), -1)
        for i in range(num_imgs):
            img = imgs[i]

            #TODO probably need to transform to some different colorspace to really get the contrast of the tracking dot
            # hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(img, lower, upper)
            # mask = cv2.inRange(blurred, lower, upper) # TODO need to define lower, upper bound for intensity in the image?
            # TODO second parameter = kernel size used to convolve image. 
            # Need to play around with that and iterations to get better performance
            if not self.mocap_cfg['use_fake_imgs']:
                mask = cv2.erode(mask, None, iterations=2)
                mask = cv2.dilate(mask, None, iterations=2)
            # TODO the second parameter here, CHAIN_APPROX_SIMPLE compresses the contour by storing only the points
            # that are absolutely necessary. For a circle, this might be redundant, as technically all points should be stored
            # The utility of this parameter is specifically for reducing the memory footprint of the contour but I suspect that
            # it will cost more time for it to try and reduce the contour footprint with not much gain since its a circle.
            # Other option is cv2.CHAIN_APPROX_NONE which stores all contour points. Might be bad memory wise but might be faster? 
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = list(contours)
            if len(contours) < num_centers:
                return None
            contours.sort(key=cv2.contourArea, reverse=True) # assumption is that dots are the biggest thing in the img
            contours = contours[:num_centers]
            img_centers = []
            for c in contours:
                M = cv2.moments(c)
                img_centers.append([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])
            centers[i] = img_centers
        return centers
    
    # TODO basically copy paste of scripts/test_bundle_adjustment.py
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
        # for i in range(self.n_cams):
        #     ext_c1w = self.env.extrinsics_cw[0]
        #     pos = x0[2*i, :].reshape((3, 1))
        #     pos_homo = np.vstack((pos, [[1.]]))
        #     print(self.env.cam_pos[i].reshape((3, 1)) - (ext_c1w @ pos_homo))
        #     assert np.allclose(self.env.cam_pos[i].reshape((3, 1)), ext_c1w @ pos_homo, atol=1e-6), f"{self.env.cam_pos[i]}, {(ext_c1w @ pos_homo), (ext_c1w @ pos_homo).shape}"
        breakpoint()
        x0 = x0[2:] # get rid of first cam
        res = utils.ba_calc_residuals(x0.flatten(), self.n_cams, self.intrinsics, obs_2d)
        breakpoint()
        res = least_squares(fun=utils.ba_calc_residuals,
                            x0=x0.flatten(),
                            # loss='huber',
                            # f_scale=4.0,
                            # ftol=2.2e-16,
                            # xtol=2.2e-16,
                            verbose=2,
                            args=(self.n_cams, self.intrinsics, obs_2d))
    
        # ---------- EVALUATE BA ----------
        gt_projections = self.env.gt_projections
        og_projections = self.env.projections
        new_projections = np.empty((self.n_cams, 3, 4))
        ext_wc1 = self.env.gt_extrinsics_wc[0]     
        for i in range(self.n_cams):
            if i == 0:
                new_projections[0] = self.intrinsics[0] @ ext_wc1
            else:
                params = res.x[6*(i-1):6*i]
                pos = params[:3].reshape((3, 1))
                rot_vec = params[3:6].reshape((3,))
                rot_mtrx = R.from_rotvec(rot_vec).as_matrix()
                ext_c1c = np.hstack((rot_mtrx, pos))
                intrinsic = self.intrinsics[i, :, :]
                new_projections[i, :, :] = intrinsic @ utils.compose_Ps(ext_c1c, ext_wc1) 
        n_eval = 10
        obs_2d = np.empty((self.n_cams * n_eval, 2), dtype=int)
        obs_3d = np.empty((3, n_eval), dtype=np.float32)
        for i in range(n_eval):
            imgs = self.read_cameras()
            pixels = self.locate_centers(imgs, 
                                            num_centers=1, 
                                            lower=200,
                                            upper=256).squeeze()
            obs_2d[self.n_cams*i:self.n_cams*(i+1)] = pixels
        gt_pred = utils.project_2d_to_3d(self.n_cams, gt_projections, obs_2d)
        og_pred = utils.project_2d_to_3d(self.n_cams, og_projections, obs_2d)
        new_pred = utils.project_2d_to_3d(self.n_cams, new_projections, obs_2d)
        print(f"L2-norm(gt_pred - new_pred): {np.linalg.norm(gt_pred - new_pred, axis=1)}")
        print(f"L2-norm(gt_pred vs og_pred): {np.linalg.norm(gt_pred - og_pred, axis=1)}")
        breakpoint()
        pos = np.empty((self.n_cams, 3))
        eulers = np.empty((self.n_cams, 3))
        for i in range(self.n_cams):
            ext = np.linalg.inv(self.intrinsics[i]) @ new_projections[i]
            R_wc = ext[:, :3]
            t_wc = ext[:, 3].reshape(3, 1)
            R_cw = R_wc.T
            t_cw = -R_cw @ t_wc
            pos[i] = t_cw.flatten()
            eulers[i] = R.from_matrix(R_cw).as_euler('xyz', degrees=False)
        self.env.cam_pos = pos
        self.env_cam_eulers = eulers
        self.env.render()
        breakpoint()
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
    
    def construct_extrinsics(self):
        # env.extrinsics are going to be in world frame
        # want to guaruntee that cam1 is the origin of the mocap frame
        # therefore, compose pose_fix as inverse of cam1 world extrinsics and multiply all other cams
        # to ensure all camera frames are relative to cam1 origin
        self.ext_cc1, self.ext_c1c = self.to_cam1(self.env.extrinsics_wc, self.env.extrinsics_cw)
        # pos_homo = np.vstack((self.env.cam_pos.T, np.ones((1, self.n_cams))))
        # ext_wc1_homo = utils.homogenize_Ps(self.env.extrinsics_wc[0])
        # ext_c1w_homo = utils.homogenize_Ps(self.env.extrinsics_cw[0])
        # for i in range(4):
        #     print(self.ext_c1c[i] @ ext_wc1_homo @ pos_homo[:, i])
        #     assert np.allclose(self.ext_c1c[i] @ ext_wc1_homo @ pos_homo[:, i], np.zeros((3, 1)))
        # for i in range(4):
        #     print(self.env.extrinsics_cw[0] @ utils.homogenize_Ps(self.ext_cc1[i]) @ np.array([0., 0., 0., 1.]).reshape((4, 1)))
        #     # assert np.allclose(self.env.extrinsics_cw[0] @ utils.homogenize_Ps(self.ext_cc1[i]) @ np.array([0., 0., 0., 1.]).reshape((4, 1)), 
        #     #                    self.env.cam_pos[i])

        breakpoint()
        # ext_homo = utils.homogenize_Ps(ext)
        # pose_fix_inv_homo = utils.homogenize_Ps(self.pose_fix_inv)
        # for i in range(4):
        #     assert np.allclose(self.env.extrinsics_wc[i], (pose_fix_inv_homo @ ext_homo[i])[:3, :])
        #     print(f"------------ cam{i} ------------")
        #     print(self.env.extrinsics_wc[i])
        #     print((self.pose_fix_inv @ ext_homo[i])[:3, :])
        # return ext
    
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

        # --------- ARCHIVE, COMPLETE NONSENSE ---------
        # transformed_ext = np.empty((self.n_cams, 3, 4), dtype=np.float32)
        # pose_fix_R = extrinsics[0][:, :3].T
        # t = extrinsics[0][:3, 3][:, np.newaxis]
        # pose_fix_t = -pose_fix_R @ t
        # pose_fix = np.hstack((pose_fix_R, pose_fix_t))
        # pose_fix_homo = utils.homogenize_Ps(pose_fix) 
        # pose_fix_inv = np.hstack((pose_fix_R.T, t))
        # breakpoint()
        # for i in range(self.n_cams):
        #     ext_i_homo = utils.homogenize_Ps(extrinsics[i])
        #     transformed_ext[i] = (pose_fix_homo @ ext_i_homo)[:3, :]
        # return transformed_ext, pose_fix, pose_fix_inv
    
    def construct_projections(self):
        self.projections = self.intrinsics @ self.extrinsics
    
    def read_cameras(self):
        feature_pt = self.env.gen_random_pt()
        while not self.env.pt_in_fovs(feature_pt) or not self.env.gen_imgs():
            feature_pt = self.env.gen_random_pt()
        self.env.render()
        fake_imgs = self.env.fake_imgs
        return fake_imgs
    
    # ---------------------------- ARCHIVE ---------------------------
    # def calibrate_camera_poses(self, filename):
    #     """
    #     Given images captured by mocap, extract features to calculate camera poses
    #     Output JSON file containing calibration information for each of the cameras in mocap system
    #     Inputs:
    #         filename (str): path to .npz file containing image data

    #     start with (num_timesteps, num_cams, resolution)
    #     1. go through and find the centers for each img                          - (num_timesteps, num_cams, num_centers, 2)
    #     2. for each timestep, establish correspondences using wand geometry      - (num_timesteps, num_cams, num_centers, 2)
    #     3. flatten across timesteps dimensions                                   - (num_timesteps * num_centers, num_cams, 2)
    #     4. Use RANSAC to randomly sample 8 points, estimate fundamental matrix between Cam 0->1, 0->2, 0->3
    #     5. Calculate essential matrix using fundamental + intrinsics
    #     6. Decompose essential to get rotation, translation up to scale
    #     7. Use this relative transformation + known wand distance to calculate scale factor
    # ---------------- THIS GIVES THE INITIAL GUESS X0 ------------------------------------------
    #     8. Refine relative transformation using BA
    #     """
    #     all_imgs = np.load(filename)['mocap_imgs'] # [num_timesteps, num_cams, resolution]
    #     num_timesteps, num_cams, _ = all_imgs.shape

    #     # step 1
    #     centers = []
    #     for i in range(num_timesteps):
    #         centers_i = self.locate_centers(imgs=all_imgs[i],
    #                                        num_centers=self.wand_params['num_centers'],
    #                                        lower=self.cfg['mask_lower'],
    #                                        upper=self.cfg['mask_upper'])
    #         if centers_i is not None:
    #             centers.append(centers_i)
    #     centers = np.array(centers)

    #     # step 2 

    

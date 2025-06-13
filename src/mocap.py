from pseyepy import Camera, cam_count
from abc import ABC, abstractmethod
import cv2
import numpy as np
import time
import src.utils as utils
from src.env import TestEnv
from scipy.optimize import least_squares

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

        self.intrinsics = self.construct_intrinsics()
        self.extrinsics = self.construct_extrinsics()
        self.projections = self.construct_projections()
    
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
        obs_3d_og = np.empty((n_obs, 3), dtype=np.float32)
        obs_3d_gt = np.empty((n_obs, 3), dtype=np.float32)
        for i in range(n_obs):
            imgs = self.read_cameras()
            pixels = self.locate_centers(imgs, 
                                         num_centers=1, 
                                         lower=200,
                                         upper=256).squeeze() 
            obs_3d_og[i] = utils.DLT(pixels, self.projections)
            obs_3d_gt[i] = utils.DLT(pixels, self.env.gt_projections)
            obs_2d[self.n_cams*i:self.n_cams*(i+1)] = pixels
        # x0_og = np.concatenate((self.projections.flatten(), 
        #                      obs_3d_og.flatten()))
        # x0_gt = np.concatenate((self.env.gt_projections.flatten(), 
        #                      obs_3d_gt.flatten()))
        interleaved_og = np.empty((8, 3), dtype=np.float32)
        interleaved_og[0::2] = self.env.cam_pos
        interleaved_og[1::2] = self.env.cam_eulers

        interleaved_gt = np.empty((8, 3), dtype=np.float32)
        interleaved_gt[0::2] = self.env.gt_cam_pos
        interleaved_gt[1::2] = self.env.gt_cam_eulers
 
        x0_og = np.concatenate((interleaved_og.flatten(),
                                obs_3d_og.flatten()))
        x0_gt = np.concatenate((interleaved_gt.flatten(),
                                obs_3d_gt.flatten()))
        og_residuals = utils.ba_calc_residuals(x0_og, self.n_cams, self.intrinsics, obs_2d)
        gt_residuals = utils.ba_calc_residuals(x0_gt, self.n_cams, self.intrinsics, obs_2d)
        breakpoint()
        res = least_squares(fun=utils.ba_calc_residuals,
                            x0=x0_og,
                            verbose=2,
                            xtol=1e-16,
                            ftol=1e-16,
                            args=(self.n_cams, self.intrinsics, obs_2d))
        breakpoint()
        # ---------- EVALUATE BA ----------
        gt_projections = self.env.gt_projections
        og_projections = self.env.projections
        new_projections = np.empty((self.n_cams, 3, 4))
        for i in range(self.n_cams):
            params = res.x[6*i:6*(i+1)]
            pos = params[:3].reshape((3, 1))
            eulers = params[3:6].reshape((3,))
            ext_wc, _ = utils.construct_extrinsics(pos, eulers)
            intrinsic = self.intrinsics[i, :, :] 
            new_projections[i, :, :] = intrinsic @ ext_wc
        # new_projections = res.x[:48].reshape((4, 3, 4))

        def project(Ps, obs_3d):
            obs_3d = np.vstack((obs_3d, np.ones((1, obs_3d.shape[1]))))             # add homo coords, (4, n_eval)
            projected_2d = Ps @ obs_3d                                              # (n_cams, 3, 4) @ (4, n_eval) = (n_cams, 3, n_eval)
            projected_2d = np.transpose(projected_2d, (2, 0, 1)).reshape(-1, 3)     # (n_eval * n_cams, 3). ordering of (2, 0, 1) is important to preserve proper interleaving
            projected_2d = projected_2d / projected_2d[:, -1][:, np.newaxis]
            projected_2d = projected_2d[:, :2]                                      # get rid of homo coords, (n_eval * n_cams, 2)
            projected_2d = np.round(projected_2d)
            return projected_2d
        n_eval = 5
        obs_3d = np.empty((3, n_eval), dtype=np.float32)
        for i in range(n_eval):
            imgs = self.read_cameras()
            obs_3d[:, i] = self.env.feature_pt

        gt_pred = project(gt_projections, obs_3d)
        og_pred = project(og_projections, obs_3d)
        new_pred = project(new_projections, obs_3d)
        breakpoint()
            

class FakeMocap(BaseMocap):
    def __init__(self, mocap_cfg, env_cfg):
        self.env = TestEnv(env_cfg)
        self.n_cams = self.env.n_cams
        self.resolution = self.env.resolution

        super().__init__(mocap_cfg)

    def construct_intrinsics(self):
        return self.env.intrinsics        
    
    def construct_extrinsics(self):
        return self.env.extrinsics_wc
    
    def construct_projections(self):
        return self.env.projections
    
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

    

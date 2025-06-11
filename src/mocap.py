from pseyepy import Camera, cam_count
import cv2
import numpy as np
import time
import src.utils as utils

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

class Mocap:
    def __init__(self,
                 cfg):
        self.cfg = cfg
        self.pseyepy_params = cfg['pseyepy_params']
        # self.wand_params = cfg['wand_params'] # TODO not needed anymore

        if self.pseyepy_params['resolution'] in ['small', 'large']:
            res = self.pseyepy_params['resolution']
            self.resolution = (320, 240) if res == 'small' else (640, 480)
            self.pseyepy_params['resolution'] = Camera.RES_SMALL if res == 'small' else Camera.RES_LARGE
        else:
            raise Exception(f"pseyepy_params.resolution not in ['small', 'large], received {self.pseyepy_params['resolution']} of type {type(self.pseyepy_params['resolution'])}")
        
        if self.cfg['use_fake_imgs']:
            self.n_cams = 4
            self.cameras = FakeCameras(fake_imgs_filepath=self.cfg['fake_imgs_filepath'])
        else:
            self.n_cams = cam_count()
            self.cameras = Camera(**self.pseyepy_params)

        self.cam_pos = np.stack([self.cfg['pos']['cam1'],
                                 self.cfg['pos']['cam2'],
                                 self.cfg['pos']['cam3'],
                                 self.cfg['pos']['cam4']], axis=0)
        self.cam_eulers = np.stack([self.cfg['eulers']['cam1'],
                                    self.cfg['eulers']['cam2'],
                                    self.cfg['eulers']['cam3'],
                                    self.cfg['eulers']['cam4']], axis=0)
        self.cam_eulers = np.radians(self.cam_eulers)
        self.construct_intrinsics()
        self.construct_projections()

    def construct_intrinsics(self):
        self.intrinsics = np.empty((4, 3, 3))
        for i, cam in enumerate(['cam1', 'cam2', 'cam3', 'cam4']):
            fx = self.cfg['intrinsics'][cam]['fx'] 
            fy = self.cfg['intrinsics'][cam]['fy']
            ox = self.cfg['intrinsics'][cam]['ox'] 
            oy = self.cfg['intrinsics'][cam]['oy']
            self.intrinsics[i, :, :] = np.array(
                [
                    [fx, 0., ox],
                    [0., fy, oy],
                    [0., 0., 1.]
                ]
            )

    def construct_projections(self):
        self.extrinsics_wc = np.empty((4, 3, 4))
        self.extrinsics_cw = np.empty((4, 3, 4))
        for i in range(self.n_cams):
            R_cw = utils.generate_rotation_matrix_from_eulers(self.cam_eulers[i]) # body -> world, +x forward, +y left, +z up
            R_wc = R_cw.T # world -> body
            # need to transform to +x left, +y up, +z forward
            # this ensures optical axis is aligned with +z enabling proper homogenous calculation
            R_wc = np.array([
                [0., 1., 0.,], 
                [0., 0., 1.], 
                [1., 0., 0.,]
            ]) @ R_wc
            t = -R_wc @ self.cam_pos[i][:, np.newaxis]
            ext_wc = np.hstack((R_wc, t))
            
            # can't directly use R_cw here because R_wc includes a camera frame transform. 
            # Therefore, to be completel correct, best to just do R_wc.T which includes that frame transform
            ext_cw = np.hstack((R_wc.T, self.cam_pos[i][:, np.newaxis])) 
            self.extrinsics_wc[i, :, :] = ext_wc
            self.extrinsics_cw[i, :, :] = ext_cw
        self.projections = self.intrinsics @ self.extrinsics_wc

    def read_cameras(self):
        imgs, timesteps = self.cameras.read()
        for img in imgs:
            img = img.copy()
            img = cv2.GaussianBlur(img, self.cfg['gauss_blur_kernel'], 0)
        return imgs, timesteps
    
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
            if not self.cfg['use_fake_imgs']:
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

    def bundle_adjustment(self):
        pass

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

    

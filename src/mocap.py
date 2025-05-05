from pseyepy import Camera, cam_count
import cv2
import numpy as np

class Mocap:
    def __init__(self,
                 cfg):
        self.cfg = cfg
        self.cam_params = cfg['camera_params']

        if self.cam_params['resolution'] in ['small', 'large']:
            res = self.cam_params['resolution']
            self.cam_params = Camera.RES_SMALL if res == 'small' else Camera.RES_LARGE 
        else:
            raise Exception(f"camera_params.resolution not in ['small', 'large], received {self.cam_params['resolution']} of type {type(self.cam_params['resolution'])}")
        
        self.num_cameras = cam_count()
        self.cameras = Camera(**self.cam_params)

    def read_cameras(self):
        imgs, timesteps = self.cameras.read()
        for img in imgs:
            img = img.copy()
            img = cv2.GaussianBlur(img, self.cfg['gauss_blur_kernel'], 0)
        return imgs, timesteps
    
    def collect_data(self,
                     num_frames,
                     save = False):
        data = np.empty((num_frames, self.num_cameras, 2))
        for i in range(num_frames):
            imgs, timesteps = self.read_cameras()
            centers = self.track_ball(imgs,
                                    lower = self.cfg['mask_lower'],
                                    upper = self.cfg['mask_upper'])
            data[i, :, :] = np.array(centers)
        if save:
            filename = f'mocap_{num_frames}_frames'
            np.save(filename, data=data)
        
    def track_ball(self, imgs, lower, upper):
        centers = []
        for i, img in enumerate(imgs):
            #TODO probably need to transform to some different colorspace to really get the contrast of the tracking dot
            # hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
            
            mask = cv2.inRange(img, lower, upper)
            # mask = cv2.inRange(blurred, lower, upper) # TODO need to define lower, upper bound for intensity in the image?
            # TODO second parameter = kernel size used to convolve image. 
            # Need to play around with that and iterations to get better performance
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            # TODO the second parameter here, CHAIN_APPROX_SIMPLE compresses the contour by storing only the points
            # that are absolutely necessary. For a circle, this might be redundant, as technically all points should be stored
            # The utility of this parameter is specifically for reducing the memory footprint of the contour but I suspect that
            # it will cost more time for it to try and reduce the contour footprint with not much gain since its a circle.
            # Other option is cv2.CHAIN_APPROX_NONE which stores all contour points. Might be bad memory wise but might be faster? 
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            center = [-1, -1]
            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                # ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]
            centers.append(center)
        return centers
    

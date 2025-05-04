from pseyepy import Camera, cam_count
import cv2
import numpy as np

class Mocap:
    def __init__(self,
                 fps = 150):
        self.cameras = c = Camera(fps=150, 
                                    resolution=Camera.RES_SMALL, 
                                    colour=True, 
                                    auto_gain = False, 
                                    auto_exposure = False,
                                    gain=63, 
                                    exposure=255)

    def track_ball(self,
                   lower,
                   upper):
        img, timestep = self.cameras.read()
        img = img.copy()
        blurred = cv2.GaussianBlur(img, (11, 11), 0)
        # hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
        #TODO probably need to transform to some different colorspace to really get the contrast of the tracking dot
        
        mask = cv2.inRange(blurred, lower, upper)
        # cv2.imshow("mask", mask)
        # cv2.waitKey(1000)
        # mask = cv2.inRange(blurred, lower, upper) # TODO need to define lower, upper bound for intensity in the image?
        # TODO second parameter = kernel size used to convolve image. 
        # Need to play around with that and iterations to get better performance
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # combined = np.hstack((hsv, mask))
        # cv2.imshow("mask", mask)
        # cv2.waitKey(1000)

        # TODO the second parameter here, CHAIN_APPROX_SIMPLE compresses the contour by storing only the points
        # that are absolutely necessary. For a circle, this might be redundant, as technically all points should be stored
        # The utility of this parameter is specifically for reducing the memory footprint of the contour but I suspect that
        # it will cost more time for it to try and reduce the contour footprint with not much gain since its a circle.
        # Other option is cv2.CHAIN_APPROX_NONE which stores all contour points. Might be bad memory wise but might be faster? 
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = (0, 0)
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            # ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return img, center
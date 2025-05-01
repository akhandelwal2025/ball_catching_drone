from pseyepy import Camera, cam_count
import cv2

class Mocap:
    def __init__(self,
                 fps = 120):
        self.cameras = Camera(fps=fps)

    def track_ball(self):
        img, timestep = self.cameras.read()
        blurred = cv2.GaussianBlur(img, (11, 11), 0)
        #TODO probably need to transform to some different colorspace to really get the contrast of the tracking dot
        
        mask = cv2.inRange(blurred, lower, upper) # TODO need to define lower, upper bound for intensity in the image?

        # TODO second parameter = kernel size used to convolve image. 
        # Need to play around with that and iterations to get better performance
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # TODO the second parameter here, CHAIN_APPROX_SIMPLE compresses the contour by storing only the points
        # that are absolutely necessary. For a circle, this might be redundant, as technically all points should be stored
        # The utility of this parameter is specifically for reducing the memory footprint of the contour but I suspect that
        # it will cost more time for it to try and reduce the contour footprint with not much gain since its a circle.
        # Other option is cv2.CHAIN_APPROX_NONE which stores all contour points. Might be bad memory wise but might be faster? 
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
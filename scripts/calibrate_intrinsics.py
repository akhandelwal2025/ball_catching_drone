from pseyepy import Camera, cam_count
import cv2
import sys
import keyboard
import numpy as np

def calibrate(c, save_file):
    # Define the checkerboard size (number of inner corners per row and column)
    # For a checkerboard with 9x6 squares, there are 8x5 internal corners
    checkerboard_size = (8, 6)
    square_size = 0.0254  # m
    objp = np.zeros((checkerboard_size[0]*checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    imgs = []
    imgpoints = []
    objpoints = []
    while len(imgs) < 10:
        frame, timestep = c.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        # Find the corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        if ret:
            # Refine corner positions to subpixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Draw and display the corners
            image_with_corners = cv2.drawChessboardCorners(image, checkerboard_size, corners_refined, ret)

            # Show image
            cv2.imshow('Checkerboard Corners', image_with_corners)
            key = cv2.waitKey(1) & 0xFF
            if ret and key != 255:
                imgs.append(frame) 
                imgpoints.append(corners)
                objpoints.append(objp)    
                print(f'saving img + corners. num saved = {len(imgs)}')
        else:
            cv2.imshow("Checkerboard Corners", image)
            cv2.waitKey(1)

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, 
                                                     imgpoints, 
                                                     gray.shape[::-1], 
                                                     None, 
                                                     None
                                )
    print(ret)
    print(f"intrinsics: {K}")
    print(f"distortion coeffs: {dist}")
    print(f"checkerboard pose: {rvecs}")

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("Total reprojection error:", mean_error / len(objpoints))

    np.savez(save_file, intrinsics=K, distortion_coeffs=dist)

if __name__ == "__main__":
    c = Camera(fps=150, 
                resolution=Camera.RES_SMALL,
                gain=63,
                exposure=255)
    save_file = input("Filename to save intrinsic matrix under: ")
    print(f"Saving under {save_file}")
    imgs = calibrate(c, save_file)

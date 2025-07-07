from pseyepy import Camera, cam_count
import cv2
import sys

c = Camera(fps=150, 
           resolution=Camera.RES_SMALL,
           gain=63,
           exposure=255)
while True:
    frame, timestep = c.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define the checkerboard size (number of inner corners per row and column)
    # For a checkerboard with 9x6 squares, there are 8x5 internal corners
    checkerboard_size = (8, 6)

    # Find the corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if ret:
        # Refine corner positions to subpixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Draw and display the corners
        image_with_corners = cv2.drawChessboardCorners(image, checkerboard_size, corners_refined, ret)
        print("here")
        # Show image
        cv2.imshow('Checkerboard Corners', image_with_corners)
        cv2.waitKey(1)
    else:
        print("Checkerboard corners not found.")
        cv2.imshow("Raw Gray Image", gray)
        cv2.waitKey(1)
 
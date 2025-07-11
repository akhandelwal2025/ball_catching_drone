import open3d as o3d
import numpy as np
import time
import cv2

deg = 0
count = 0
vis = o3d.visualization.Visualizer()
vis.create_window()
fake_imgs = np.load('data/fake_imgs_100.npz')['fake_imgs']

frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
vis.add_geometry(frame)
start = time.time()
try:
    while True:
        R_x = np.array([
            [1., 0., 0.,],
            [0., np.cos(-np.radians(deg)), -np.sin(-np.radians(deg))],
            [0., np.sin(-np.radians(deg)), np.cos(-np.radians(deg))]
        ])
        deg = (deg + 1) % 360
        frame.rotate(R_x, center=(0, 0, 0))
        vis.update_geometry(frame)
        vis.poll_events()
        vis.update_renderer()

        cv2.namedWindow("Camera 1")
        cv2.namedWindow("Camera 2")
        cv2.namedWindow("Camera 3")
        cv2.namedWindow("Camera 4")

        cv2.moveWindow("Camera 1", 650, 520)
        cv2.moveWindow("Camera 2", 0, 520)
        cv2.moveWindow("Camera 3", 0, 0)
        cv2.moveWindow("Camera 4", 650, 0)

        imgs = fake_imgs[deg % 100]
        cv2.imshow("Camera 1", imgs[0])
        cv2.imshow("Camera 2", imgs[1])
        cv2.imshow("Camera 3", imgs[2])
        cv2.imshow("Camera 4", imgs[3])

        cv2.waitKey(1)
        count += 1
except:
    end = time.time()
    elapsed = end - start
    print(f"{count} frames in {elapsed} seconds -> {count / elapsed} fps")
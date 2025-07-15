import open3d as o3d
import numpy as np
import time
import cv2


def R_x(theta):
    return np.array([
    [1., 0., 0.,],
    [0., np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
    [0., np.sin(np.radians(theta)), np.cos(np.radians(theta))]
])

def R_y(theta):
    return np.array([
        [np.cos(np.radians(theta)), 0., np.sin(np.radians(theta))],
        [0., 1., 0.,],
        [-np.sin(np.radians(theta)), 0., np.cos(np.radians(theta))]
    ])

def R_z(theta):
    return np.array([
        [np.cos(np.radians(theta)), -np.sin(np.radians(theta)), 0.],
        [np.sin(np.radians(theta)), np.cos(np.radians(theta)), 0.],
        [0., 0., 1.]
    ])

count = 0
vis = o3d.visualization.Visualizer()
vis.create_window()
fake_imgs = np.load('data/fake_imgs_100.npz')['fake_imgs']
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0., 0., 0.])
vis.add_geometry(origin)
# origin.rotate(R_x(-90), center=(0, 0, 0))
R = origin.get_rotation_matrix_from_xzy([np.radians(-90), 0., 0.])
origin.rotate(R, center=(0, 0, 0))
vis.update_geometry(origin)

frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
vis.add_geometry(frame)
# frame.rotate(R_y(45), center=(0, 0, 0))
# vis.update_geometry(frame)
# frame.rotate(R_z(25), center=(0, 0, 0))
R = origin.get_rotation_matrix_from_xzy([np.radians(-90), np.radians(45), np.radians(25)])
frame.rotate(R, center=(0, 0, 0))

new_x = R_x(-90) @ np.array([1, 0, 0])
new_y = R_x(-90) @ np.array([0, 1, 0])
# print(new_x, new_y)
# # 4. Translate along new x and y directions
translation_vector = 0. * new_x + 0. * new_y  # customize this
frame.translate(translation_vector)
vis.update_geometry(frame)

while True:
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.01)
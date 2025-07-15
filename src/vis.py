import matplotlib.pyplot as plt
import numpy as np
import src.utils as utils
import cv2
import open3d as o3d

class Vis():
    def __init__(self,
                 cfg,
                 cam_pos,
                 cam_eulers):
        self.cfg = cfg

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.pcd = o3d.geometry.PointCloud() # used to visualize points
        
        self.draw_axes(origins=np.array([[0., 0., 0.,]]),
                       eulers=np.array([[0., 0., 0.,]]))
        self.draw_axes(cam_pos, cam_eulers)
        self.vis.add_geometry(self.pcd)

    def draw_axes(self,
                  origins: np.ndarray,
                  eulers: np.ndarray):
        """
        draw a coordinate axis given an origin point and eulers defining the rotation
        Inputs:
            origin: np.ndarray - position of the coordinate frames origin. shape = (N, 3), where N = num frames
            eulers: np.ndarray - rotation of the coordinate frame. shape = (N, 3), where N = num frames
        """
        origins = origins / 2.0
        N = origins.shape[0]
        # needed to rotate the frame 90 deg about x-axis to get expected standard coordinate frame
        R_x = np.array([
            [1., 0., 0.,],
            [0., np.cos(-np.pi/2), -np.sin(-np.pi/2)],
            [0., np.sin(-np.pi/2), np.cos(-np.pi/2)]
        ])
        for i in range(N):
            origin = origins[i]
            euler = eulers[i]
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            R = frame.get_rotation_matrix_from_xzy([np.radians(-90), euler[2], euler[1]])
            frame.rotate(R, center=[0., 0., 0.,])

            new_x = R_x @ np.array([1, 0, 0])
            new_y = R_x @ np.array([0, 1, 0])
            new_z = R_x @ np.array([0, 0, 1])
            translation = origin[0] * new_x + origin[1] * new_y + origin[2] * new_z
            frame.translate(translation)
            self.vis.add_geometry(frame)
            self.vis.poll_events()
            self.vis.update_renderer()

    def render(self,
               centers,
               imgs,
               pt_3d):
        
        R_x = np.array([
            [1., 0., 0.,],
            [0., np.cos(-np.pi/2), -np.sin(-np.pi/2)],
            [0., np.sin(-np.pi/2), np.cos(-np.pi/2)]
        ])
        new_x = R_x @ np.array([1, 0, 0])
        new_y = R_x @ np.array([0, 1, 0])
        new_z = R_x @ np.array([0, 0, 1])
        pt_3d = pt_3d / 2
        translation = (pt_3d[0, 0] * new_x + pt_3d[0, 1] * new_y + pt_3d[0, 2] * new_z)[np.newaxis, :]
        print(f"translations: {translation.shape}")
        self.pcd.points = o3d.utility.Vector3dVector(translation)
        self.pcd.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]])
        self.vis.update_geometry(self.pcd)

        self.vis.poll_events()
        self.vis.update_renderer()

        # render each of the images
        cv2.namedWindow("Camera 1")
        cv2.namedWindow("Camera 2")
        cv2.namedWindow("Camera 3")
        cv2.namedWindow("Camera 4")

        # Move the windows to a 2x2 grid layout on screen (optional)
        cv2.moveWindow("Camera 1", 445, 325)
        cv2.moveWindow("Camera 2", 0, 325)
        cv2.moveWindow("Camera 3", 0, 0)
        cv2.moveWindow("Camera 4", 445, 0)

        if centers is not None:
            n_centers = centers.shape[0] // 4
            for c in range(n_centers):
                for i in range(4):
                    img = imgs[i]
                    img = cv2.circle(img, centers[4 * c + i], radius=3, color=[0, 0, 255])
                    imgs[i] = img
        cv2.imshow("Camera 1", imgs[0])
        cv2.imshow("Camera 2", imgs[1])
        cv2.imshow("Camera 3", imgs[2])
        cv2.imshow("Camera 4", imgs[3])

        cv2.waitKey(1)
# -------------- ARCHIVE --------------
# class Vis():
#     def __init__(self,
#                  cfg, 
#                  xlim = (-2, 10),
#                  ylim = (-2, 10),
#                  zlim = (0, 12)):
#         self.cfg = cfg

#         self.fig = plt.figure()
#         self.gs = self.fig.add_gridspec(2, 3)

#         self.ax3D = self.fig.add_subplot(self.gs[:, 0], projection='3d')
#         self.ax3D.set_xlim(xlim)  # Set x-axis limits to a range around zero
#         self.ax3D.set_ylim(ylim)  # Set y-axis limits to a range around zero
#         self.ax3D.set_zlim(zlim)

#         self.ax2d_1 = self.fig.add_subplot(self.gs[0, 1])
#         self.ax2d_2 = self.fig.add_subplot(self.gs[0, 2])
#         self.ax2d_3 = self.fig.add_subplot(self.gs[1, 1])
#         self.ax2d_4 = self.fig.add_subplot(self.gs[1, 2])

#         plt.ion()
    
#     def draw_axes(self,
#                   origin: np.ndarray,
#                   eulers: np.ndarray):
#         N = origin.shape[0]
#         rotation = np.empty((N, 3, 3))
#         rotation = np.apply_along_axis(func1d=utils.generate_rotation_matrix_from_eulers, 
#                                        axis=1,
#                                        arr=eulers)
#         for i in range(N):
#             self.ax3D.quiver(origin[i, 0], origin[i, 1], origin[i, 2], *rotation[i, :, 0], color='red', pivot='tail', length=2, normalize=True)
#             self.ax3D.quiver(origin[i, 0], origin[i, 1], origin[i, 2], *rotation[i, :, 1], color='green', pivot='tail', length=2, normalize=True)
#             self.ax3D.quiver(origin[i, 0], origin[i, 1], origin[i, 2], *rotation[i, :, 2], color='blue', pivot='tail', length=2, normalize=True)

#     def draw_cones(self,
#                    origins: np.ndarray,
#                    eulers: np.ndarray):
#         theta = np.radians(self.cfg['cam_FOV_half_angle'])
#         L = self.cfg['cone_length']
#         r = L * np.tan(theta) # Compute base radius from FOV and length

#         # generate cone aligned with +z
#         u = np.linspace(0, 2*np.pi, 100)
#         v = np.linspace(0, 1, 80)
#         u, v = np.meshgrid(u, v)
#         x = v*r*np.cos(u)
#         y = v*r*np.sin(u)
#         z = v*L
#         pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=0)

#         # cone by default is oriented around +z. want to rotate so that cone is oriented along +x instead
#         orient_along_x = np.array([
#             [0., 0., 1.],
#             [0., 1., 0.],
#             [-1., 0., 0.]
#         ])
#         pts = orient_along_x @ pts
#         pts = np.vstack((pts, np.ones((1, pts.shape[1])))) # add homogenous coordinates

#         for i in range(len(origins)):
#             origin = origins[i][:, np.newaxis]
#             rot_mtrx = utils.generate_rotation_matrix_from_eulers(eulers[i])
#             extrinsic = np.vstack((np.hstack((rot_mtrx, origin)), 
#                            np.array([[0., 0., 0., 1.]])))
#             transformed_pts = extrinsic @ pts
#             self.ax3D.plot_surface(transformed_pts[0, :].reshape(x.shape), 
#                                  transformed_pts[1, :].reshape(y.shape), 
#                                  transformed_pts[2, :].reshape(z.shape), 
#                                  color='black', 
#                                  alpha=0.1)
            
#     def render(self,
#                imgs: np.ndarray,
#                frames_3d: dict,
#                centers: np.ndarray = None,
#                feature_pts: np.ndarray = None):
#         """
#         Render all images and 3D camera frames
#         Inputs:
#             imgs: np.ndarray - camera images. shape = (4, res)
#             frames_3d: dict - all camera frames to render in 3D.
#                               format is as follows, key = string of color to render point with, value = tuple(pos, eulers, draw_fov)
#                               eulers = (1, 3), pos = (1, 3), draw_fov = bool
#             feature_pts: np.ndarray - all feature pts to render in 3D view. shape = (N, 3) where N = num feature points
#         """
#         # self.ax3D.cla()
#         # self.ax2d_1.cla()
#         # self.ax2d_2.cla()
#         # self.ax2d_3.cla()
#         # self.ax2d_4.cla()

#         # # render all frames
#         # for color, params in frames_3d.items():
#         #     pos, eulers, draw_fov = params
#         #     self.ax3D.scatter(pos[0], pos[1], pos[2], color=color, s=30)
#         #     self.draw_axes(pos, eulers)
#         #     if draw_fov:
#         #         self.draw_cones(pos, eulers)

#         # # render feature points
#         # if feature_pts is not None:
#         #     self.ax3D.scatter(feature_pts[:, 0], feature_pts[:, 1], feature_pts[:, 2], color='black', s=30)

#         # render each of the images
#         # Create named windows
#         cv2.namedWindow("Camera 1")
#         cv2.namedWindow("Camera 2")
#         cv2.namedWindow("Camera 3")
#         cv2.namedWindow("Camera 4")

#         # Move the windows to a 2x2 grid layout on screen (optional)
#         # cv2.moveWindow("Camera 1", 0, 0)
#         # cv2.moveWindow("Camera 2", 650, 0)
#         # cv2.moveWindow("Camera 3", 0, 520)
#         # cv2.moveWindow("Camera 4", 650, 520)
#         cv2.moveWindow("Camera 1", 650, 520)
#         cv2.moveWindow("Camera 2", 0, 520)
#         cv2.moveWindow("Camera 3", 0, 0)
#         cv2.moveWindow("Camera 4", 650, 0)

#         # Show the images in separate windows
        
#         if centers is not None:
#             n_centers = centers.shape[0] // 4
#             for c in range(n_centers):
#                 for i in range(4):
#                     img = imgs[i]
#                     img = cv2.circle(img, centers[4 * c + i], radius=3, color=[0, 0, 255])
#                     imgs[i] = img
#         cv2.imshow("Camera 1", imgs[0])
#         cv2.imshow("Camera 2", imgs[1])
#         cv2.imshow("Camera 3", imgs[2])
#         cv2.imshow("Camera 4", imgs[3])

#         cv2.waitKey(1)
#         # cv2.destroyAllWindows()
#         # self.ax2d_1.imshow(imgs[2, :, :], cmap='gray')
#         # self.ax2d_2.imshow(imgs[3, :, :], cmap='gray')
#         # self.ax2d_3.imshow(imgs[1, :, :], cmap='gray')
#         # self.ax2d_4.imshow(imgs[0, :, :], cmap='gray')

#         # # set axis labels + titles
#         # self.ax3D.set_xlabel('x')
#         # self.ax3D.set_ylabel('y')
#         # self.ax2d_1.set_title("Cam3")
#         # self.ax2d_2.set_title("Cam4")
#         # self.ax2d_3.set_title("Cam2")
#         # self.ax2d_4.set_title("Cam1")

#         # plt.show()

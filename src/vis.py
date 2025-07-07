import matplotlib.pyplot as plt
import numpy as np
import src.utils as utils

class Vis():
    def __init__(self,
                 cfg, 
                 xlim = (-2, 10),
                 ylim = (-2, 10),
                 zlim = (0, 12)):
        self.cfg = cfg

        self.fig = plt.figure()
        self.gs = self.fig.add_gridspec(2, 3)

        self.ax3D = self.fig.add_subplot(self.gs[:, 0], projection='3d')
        self.ax3D.set_xlim(xlim)  # Set x-axis limits to a range around zero
        self.ax3D.set_ylim(ylim)  # Set y-axis limits to a range around zero
        self.ax3D.set_zlim(zlim)

        self.ax2d_1 = self.fig.add_subplot(self.gs[0, 1])
        self.ax2d_2 = self.fig.add_subplot(self.gs[0, 2])
        self.ax2d_3 = self.fig.add_subplot(self.gs[1, 1])
        self.ax2d_4 = self.fig.add_subplot(self.gs[1, 2])

        plt.ion()
    
    def draw_axes(self,
                  origin: np.ndarray,
                  eulers: np.ndarray):
        N = origin.shape[0]
        rotation = np.empty((N, 3, 3))
        rotation = np.apply_along_axis(func1d=utils.generate_rotation_matrix_from_eulers, 
                                       axis=1,
                                       arr=eulers)
        for i in range(N):
            self.ax3D.quiver(origin[i, 0], origin[i, 1], origin[i, 2], *rotation[i, :, 0], color='red', pivot='tail', length=2, normalize=True)
            self.ax3D.quiver(origin[i, 0], origin[i, 1], origin[i, 2], *rotation[i, :, 1], color='green', pivot='tail', length=2, normalize=True)
            self.ax3D.quiver(origin[i, 0], origin[i, 1], origin[i, 2], *rotation[i, :, 2], color='blue', pivot='tail', length=2, normalize=True)

    def draw_cones(self,
                   origins: np.ndarray,
                   eulers: np.ndarray):
        theta = np.radians(self.cfg['cam_FOV_half_angle'])
        L = self.cfg['cone_length']
        r = L * np.tan(theta) # Compute base radius from FOV and length

        # generate cone aligned with +z
        u = np.linspace(0, 2*np.pi, 100)
        v = np.linspace(0, 1, 80)
        u, v = np.meshgrid(u, v)
        x = v*r*np.cos(u)
        y = v*r*np.sin(u)
        z = v*L
        pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=0)

        # cone by default is oriented around +z. want to rotate so that cone is oriented along +x instead
        orient_along_x = np.array([
            [0., 0., 1.],
            [0., 1., 0.],
            [-1., 0., 0.]
        ])
        pts = orient_along_x @ pts
        pts = np.vstack((pts, np.ones((1, pts.shape[1])))) # add homogenous coordinates

        for i in range(len(origins)):
            origin = origins[i][:, np.newaxis]
            rot_mtrx = utils.generate_rotation_matrix_from_eulers(eulers[i])
            extrinsic = np.vstack((np.hstack((rot_mtrx, origin)), 
                           np.array([[0., 0., 0., 1.]])))
            transformed_pts = extrinsic @ pts
            self.ax3D.plot_surface(transformed_pts[0, :].reshape(x.shape), 
                                 transformed_pts[1, :].reshape(y.shape), 
                                 transformed_pts[2, :].reshape(z.shape), 
                                 color='black', 
                                 alpha=0.1)
            
    def render(self,
               imgs: np.ndarray,
               frames_3d: dict,
               feature_pts: np.ndarray):
        """
        Render all images and 3D camera frames
        Inputs:
            imgs: np.ndarray - camera images. shape = (4, res)
            frames_3d: dict - all camera frames to render in 3D.
                              format is as follows, key = string of color to render point with, value = tuple(pos, eulers, draw_fov)
                              eulers = (1, 3), pos = (1, 3), draw_fov = bool
            feature_pts: np.ndarray - all feature pts to render in 3D view. shape = (N, 3) where N = num feature points
        """
        self.ax3D.cla()
        self.ax2d_1.cla()
        self.ax2d_2.cla()
        self.ax2d_3.cla()
        self.ax2d_4.cla()

        # render all frames
        for color, params in frames_3d.items():
            pos, eulers, draw_fov = params
            self.ax3D.scatter(pos[0], pos[1], pos[2], color=color, s=30)
            self.draw_axes(pos, eulers)
            if draw_fov:
                self.draw_cones(pos, eulers)

        # render feature points
        self.ax3D.scatter(feature_pts[:, 0], feature_pts[:, 1], feature_pts[:, 2], color='black', s=30)

        # render each of the images
        self.ax2d_1.imshow(imgs[0, :, :], cmap='gray')
        self.ax2d_2.imshow(imgs[1, :, :], cmap='gray')
        self.ax2d_3.imshow(imgs[2, :, :], cmap='gray')
        self.ax2d_4.imshow(imgs[3, :, :], cmap='gray')

        # set axis labels + titles
        self.ax3D.set_xlabel('x')
        self.ax3D.set_ylabel('y')
        self.ax2d_1.set_title("Cam1")
        self.ax2d_2.set_title("Cam2")
        self.ax2d_3.set_title("Cam3")
        self.ax2d_4.set_title("Cam4")

        plt.show()
        plt.pause(0.005)
import numpy as np
import matplotlib.pyplot as plt
import src.utils as utils

class TestEnv():
    def __init__(self, cfg):
        self.cfg = cfg

        # start at pos, orientation specified by cfg
        self.cam_pos = np.stack([self.cfg['pos']['cam1'],
                                 self.cfg['pos']['cam2'],
                                 self.cfg['pos']['cam3'],
                                 self.cfg['pos']['cam4']], axis=0)
        self.cam_eulers = np.stack([self.cfg['eulers']['cam1'],
                                    self.cfg['eulers']['cam2'],
                                    self.cfg['eulers']['cam3'],
                                    self.cfg['eulers']['cam4']], axis=0)
        self.cam_eulers = self.cam_eulers * (np.pi / 180.) # deg to rad conversion

        # gt cam pos, orientation are generated
        # if 'use_lookat' is True, then cam orientation is set to point at a target location
        # otherwise, the orientation is randomized according to the std
        self.gt_cam_pos = np.random.normal(loc=self.cam_pos, scale=self.cfg['pos']['std'])
        if self.cfg['use_lookat']:
            self.gt_cam_eulers = utils.lookat(origin=self.gt_cam_pos,
                                              target=np.array(self.cfg['lookat'])[np.newaxis, :], # newaxis enables broadcasting
                                              up=np.array([0., 0., 1.])[np.newaxis, :],
                                              return_eulers=True) 
        else:
            self.gt_cam_eulers = self.gt_cam_eulers * (np.pi / 180.)            

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(-2, 10)  # Set x-axis limits to a range around zero
        self.ax.set_ylim(-2, 10)  # Set y-axis limits to a range around zero
        self.ax.set_zlim(0, 12)
        if self.cfg['use_lookat']:
            self.ax.scatter(self.cfg['lookat'][0], self.cfg['lookat'][1], self.cfg['lookat'][2], 'yellow')
    
    def draw_axes(self,
                  origin: np.ndarray,
                  eulers: np.ndarray,
                  axis_color: str):
        N = origin.shape[0]
        rotation = np.empty((N, 3, 3))
        rotation = np.apply_along_axis(func1d=utils.generate_rotation_matrix_from_eulers, 
                                       axis=1,
                                       arr=eulers)
        for i in range(N):
            self.ax.quiver(origin[i, 0], origin[i, 1], origin[i, 2], *rotation[i, :, 0], color='red', pivot='tail', length=2, normalize=True)
            self.ax.quiver(origin[i, 0], origin[i, 1], origin[i, 2], *rotation[i, :, 1], color='green', pivot='tail', length=2, normalize=True)
            self.ax.quiver(origin[i, 0], origin[i, 1], origin[i, 2], *rotation[i, :, 2], color='blue', pivot='tail', length=2, normalize=True)

    def draw_cones(self,
                   origins: np.ndarray,
                   eulers: np.ndarray):
        theta = np.radians(30)      # Half angle of FOV
        L = 5.0                     # Length of the cone
        r = L * np.tan(theta)       # Compute base radius from FOV and length

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
            P = np.vstack((np.hstack((rot_mtrx, origin)), 
                           np.array([[0., 0., 0., 1.]])))
            transformed_pts = P @ pts
            self.ax.plot_surface(transformed_pts[0, :].reshape(x.shape), 
                                 transformed_pts[1, :].reshape(y.shape), 
                                 transformed_pts[2, :].reshape(z.shape), 
                                 color='black', 
                                 alpha=0.1)

    def render(self):
        # render gt_cams
        self.ax.scatter(self.gt_cam_pos[:, 0], self.gt_cam_pos[:, 1], self.gt_cam_pos[:, 2], color='green', s=30)
        self.draw_axes(self.gt_cam_pos, self.gt_cam_eulers, 'green')

        # render current cams
        self.ax.scatter(self.cam_pos[:, 0], self.cam_pos[:, 1], self.cam_pos[:, 2], color='red', s=30)
        self.draw_axes(self.cam_pos, self.cam_eulers, 'red')

        self.draw_cones(self.gt_cam_pos, self.gt_cam_eulers)
        plt.show()

        

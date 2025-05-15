import numpy as np
import matplotlib.pyplot as plt
import src.utils as utils

class TestEnv():
    def __init__(self, cfg):
        self.cfg = cfg

        # start with pos specified by cfg. target cam pos is something slightly off axis from that
        self.cam_pos = np.stack([self.cfg['pos']['cam1'],
                                 self.cfg['pos']['cam2'],
                                 self.cfg['pos']['cam3'],
                                 self.cfg['pos']['cam4']], axis=0)
        self.gt_cam_pos = np.random.normal(loc=self.cam_pos, scale=self.cfg['pos']['std'])

        if self.cfg['use_lookat']:
        else:
            self.cam_eulers = np.stack([self.cfg['eulers']['cam1'],
                                        self.cfg['eulers']['cam2'],
                                        self.cfg['eulers']['cam3'],
                                        self.cfg['eulers']['cam4']], axis=0)
            self.cam_eulers = self.cam_eulers * (np.pi / 180.) # deg to rad conversion
            
        self.randomize_cams()
        self.gt_cam_eulers = self.gt_cam_eulers * (np.pi / 180.)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(-2, 10)  # Set x-axis limits to a range around zero
        self.ax.set_ylim(-2, 10)  # Set y-axis limits to a range around zero
        self.ax.set_zlim(0, 12)

    def randomize_cams(self):
        self.gt_cam_pos = np.random.normal(loc=self.cam_pos, scale=self.cfg['pos']['std'])
        self.gt_cam_eulers = np.random.normal(loc=self.cam_eulers, scale=self.cfg['eulers']['std'])
    
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
            self.ax.quiver(origin[i, 0], origin[i, 1], origin[i, 2], *rotation[i, :, 0], color=axis_color, pivot='tail', length=2, normalize=True)
            self.ax.quiver(origin[i, 0], origin[i, 1], origin[i, 2], *rotation[i, :, 1], color=axis_color, pivot='tail', length=2, normalize=True)
            self.ax.quiver(origin[i, 0], origin[i, 1], origin[i, 2], *rotation[i, :, 2], color=axis_color, pivot='tail', length=2, normalize=True)

    def render(self):
        # render gt_cams
        self.ax.scatter(self.gt_cam_pos[:, 0], self.gt_cam_pos[:, 1], self.gt_cam_pos[:, 2], color='green')
        self.draw_axes(self.gt_cam_pos, self.gt_cam_eulers, 'green')

        # render current cams
        self.ax.scatter(self.cam_pos[:, 0], self.cam_pos[:, 1], self.cam_pos[:, 2], color='red')
        self.draw_axes(self.cam_pos, self.cam_eulers, 'red')
        plt.show()

        

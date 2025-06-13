import numpy as np
import matplotlib.pyplot as plt
import src.utils as utils

class TestEnv():
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_cams = 4
        self.resolution = self.cfg['img_resolution']

        # start at pos, orientation specified by cfg
        self.feature_pt = self.gen_random_pt()
        self.cam_pos = np.stack([self.cfg['pos']['cam1'],
                                 self.cfg['pos']['cam2'],
                                 self.cfg['pos']['cam3'],
                                 self.cfg['pos']['cam4']], axis=0)
        if self.cfg['use_lookat']:
            self.cam_eulers = utils.lookat(origin=self.cam_pos,
                                              target=np.array(self.cfg['lookat'])[np.newaxis, :], # newaxis enables broadcasting
                                              up=np.array([0., 0., 1.])[np.newaxis, :],
                                              return_eulers=True)
        else:
            self.cam_eulers = np.stack([self.cfg['eulers']['cam1'],
                                        self.cfg['eulers']['cam2'],
                                        self.cfg['eulers']['cam3'],
                                        self.cfg['eulers']['cam4']], axis=0)
            self.cam_eulers = np.radians(self.cam_eulers)
        self.construct_intrinsics()
        
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
            self.gt_cam_eulers = np.radians(self.gt_cam_eulers) #TODO wtf is this            
        self.construct_projections()
        
        self.fig = plt.figure()
        self.gs = self.fig.add_gridspec(2, 3)
        self.ax3D = self.fig.add_subplot(self.gs[:, 0], projection='3d')
        self.ax3D.set_xlim(-2, 10)  # Set x-axis limits to a range around zero
        self.ax3D.set_ylim(-2, 10)  # Set y-axis limits to a range around zero
        self.ax3D.set_zlim(0, 12)

        self.ax2d_1 = self.fig.add_subplot(self.gs[0, 1])
        self.ax2d_2 = self.fig.add_subplot(self.gs[0, 2])
        self.ax2d_3 = self.fig.add_subplot(self.gs[1, 1])
        self.ax2d_4 = self.fig.add_subplot(self.gs[1, 2])

        plt.ion()

    def construct_intrinsics(self):
        self.intrinsics = np.empty((4, 3, 3))
        for i, cam in enumerate(['cam1', 'cam2', 'cam3', 'cam4']):
            fx = self.cfg['intrinsics'][cam]['fx'] 
            fy = self.cfg['intrinsics'][cam]['fy']
            ox = self.cfg['intrinsics'][cam]['ox'] 
            oy = self.cfg['intrinsics'][cam]['oy']
            self.intrinsics[i, :, :] = utils.construct_intrinsics(fx, fy, ox, oy)

    def construct_projections(self):
        # construct for est cam locations first
        self.extrinsics_wc = np.empty((4, 3, 4))
        self.extrinsics_cw = np.empty((4, 3, 4))
        for i in range(self.n_cams):
            ext_wc, ext_cw = utils.construct_extrinsics(pos=self.cam_pos[i][:, np.newaxis],
                                                        eulers=self.cam_eulers[i])
            self.extrinsics_wc[i, :, :] = ext_wc
            self.extrinsics_cw[i, :, :] = ext_cw
        self.projections = self.intrinsics @ self.extrinsics_wc

        # repeat for ground truth
        self.gt_extrinsics_wc = np.empty((4, 3, 4))
        self.gt_extrinsics_cw = np.empty((4, 3, 4))
        for i in range(self.n_cams):
            ext_wc, ext_cw = utils.construct_extrinsics(pos=self.gt_cam_pos[i][:, np.newaxis],
                                                        eulers=self.gt_cam_eulers[i])
            self.gt_extrinsics_wc[i, :, :] = ext_wc
            self.gt_extrinsics_cw[i, :, :] = ext_cw
        self.gt_projections = self.intrinsics @ self.gt_extrinsics_wc

    def gen_imgs(self):
        w, h = self.resolution[0], self.resolution[1]
        self.fake_imgs = np.empty((4, h, w))
        self.projected_pts = np.empty((4, 2))
        feature_pt_homo = np.vstack((self.feature_pt[:, np.newaxis], np.ones((1, 1))))
        DIRS = {
            (-1, -1), (0, -1), (1, -1),
            (-1, 0),           (1, 0),
            (-1, 1),  (0, 1),  (1, 1)
        }
        
        def valid(y, x):
            return (0 <= y < h) and (0 <= x < w)
     
        for i in range(self.n_cams):
            P = self.gt_projections[i]
            projected_pt = (P @ feature_pt_homo).squeeze()
            projected_pt /= projected_pt[-1]
            projected_pt = projected_pt[:2]
            projected_pt = np.round(projected_pt).astype(int)
            self.projected_pts[i, :] = projected_pt
            x, y = projected_pt[0], projected_pt[1]
            print(y, x)
            if valid(y, x):
                fake_img = np.zeros((h, w))
                fake_img[y, x] = 255
                for dx, dy in DIRS:
                    if valid(y+dy, x+dx):
                        fake_img[y+dy, x+dx] = 255
                self.fake_imgs[i] = fake_img
            else:
                return False
        return True

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

    def gen_random_pt(self):
        bounds = np.full((3,), self.cfg['point_gen']['radius'])
        delta = np.random.uniform(low=-bounds,
                                  high=bounds)
        self.feature_pt = np.array(self.cfg['point_gen']['center']) + delta
        return self.feature_pt
    
    def pt_in_fovs(self,
                   pt):
        """
        Given a pt, check if it is in the FOV of all cameras
        Inputs:
            pt: np.ndarray - shape = (3,)
        Output True if pt can be seen by all cameras, otherwise False
        """
        for i in range(self.n_cams):
            origin = self.gt_cam_pos[i]
            R = utils.generate_rotation_matrix_from_eulers(self.gt_cam_eulers[i])
            forward_dir_wf = R @ np.array([1., 0., 0.])
            forward_dir_wf = forward_dir_wf / np.linalg.norm(forward_dir_wf)
            unit_dir = (pt-origin)/np.linalg.norm(pt-origin)
            theta = np.arccos(np.dot(unit_dir, forward_dir_wf))
            if theta > np.radians(self.cfg['cam_FOV_half_angle']):
                return False
        return True

    def render(self):
        self.ax3D.cla()
        self.ax2d_1.cla()
        self.ax2d_2.cla()
        self.ax2d_3.cla()
        self.ax2d_4.cla()

        # render gt_cams
        self.ax3D.scatter(self.gt_cam_pos[:, 0], self.gt_cam_pos[:, 1], self.gt_cam_pos[:, 2], color='green', s=30)
        self.draw_axes(self.gt_cam_pos, self.gt_cam_eulers, 'green')

        # render current cams
        # self.ax3D.scatter(self.cam_pos[:, 0], self.cam_pos[:, 1], self.cam_pos[:, 2], color='red', s=30)
        # self.draw_axes(self.cam_pos, self.cam_eulers, 'red')

        # render FOV cones
        if self.cfg['draw_FOV']:
            self.draw_cones(self.gt_cam_pos, self.gt_cam_eulers)
        
        # render lookat point
        if self.cfg['use_lookat']:
            self.ax3D.scatter(self.cfg['lookat'][0], self.cfg['lookat'][1], self.cfg['lookat'][2])
        
        # render feature point
        self.ax3D.scatter(self.feature_pt[0], self.feature_pt[1], self.feature_pt[2], color='red', s=30)

        # render each of the images
        self.ax2d_1.imshow(self.fake_imgs[0], cmap='gray')
        self.ax2d_2.imshow(self.fake_imgs[1], cmap='gray')
        self.ax2d_3.imshow(self.fake_imgs[2], cmap='gray')
        self.ax2d_4.imshow(self.fake_imgs[3], cmap='gray')

        # set axis labels + titles
        self.ax3D.set_xlabel('x')
        self.ax3D.set_ylabel('y')
        self.ax2d_1.set_title("Cam1")
        self.ax2d_2.set_title("Cam2")
        self.ax2d_3.set_title("Cam3")
        self.ax2d_4.set_title("Cam4")

        plt.show()
        plt.pause(0.01)
        

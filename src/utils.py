import numpy as np
from numpy import cos as c
from numpy import sin as s
from scipy.spatial.transform import Rotation as R

def extract_eulers_from_rotation_matrix_XYZ(rot_mtrx):
    """
    Given a rotation matrix R, extract eulers assuming XYZ rotation
    Inputs:
        R: np.ndarray - rotation matrix
    """
    # theta_y = np.arcsin(-rot_mtrx[2, 0])
    # theta_x = np.arctan2(rot_mtrx[2, 1] / np.cos(theta_y), rot_mtrx[2, 2] / np.cos(theta_y))
    # theta_z = np.arctan2(rot_mtrx[1, 0] / np.cos(theta_y), rot_mtrx[0, 0] / np.cos(theta_y))
    # # theta_x = np.arctan2(rot_mtrx[2, 1], rot_mtrx[2, 2])
    # # theta_y = np.arctan2(-rot_mtrx[2, 0], np.sqrt(rot_mtrx[0, 0]**2 + rot_mtrx[1, 0]**2))
    # # theta_z = np.arctan2(rot_mtrx[1, 0], rot_mtrx[0, 0])
    # eulers = np.array([theta_x, theta_y, theta_z])
    # breakpoint()
    # assert np.allclose(R.from_matrix(rot_mtrx).as_euler('xyz'), eulers)
    # return eulers
    return R.from_matrix(rot_mtrx).as_euler('xyz')

def lookat(origin, target, up, return_eulers=True):
    """
    Given a target pos, calculate the rotation needed to orient the body in the direction of the target
    Inputs:
        origin: np.ndarray - (N, 3) matrix indicating cam positions
        target: np.ndarray - (N, 3) matrix indicating target positions
        up: np.ndarray - (1, 3) indicating world up (i.e. +Z)
        return_eulers: bool - if True, extract eulers from rotation matrix and return those. Otherwise, return rotation matrix
            *Note: eulers are calculated assuming a XYZ rotation
    Outputs:
        if return_eulers == True, return eulers
        if return_eulers == False, return R
            *Note: rotation is from body -> world
    """
    assert np.allclose(np.linalg.norm(up), 1.0), "up vector is not normalized"
    forward = (target-origin) / np.linalg.norm(target-origin, axis=1, keepdims=True)
    left = np.cross(up, forward)
    left = left / np.linalg.norm(left, axis=1, keepdims=True)
    body_up = np.cross(forward, left)
    body_up = body_up / np.linalg.norm(body_up, axis=1, keepdims=True)

    # Note: R here is body -> world
    #       R.t then is world -> body
    R = np.stack((forward, left, body_up), axis=-1) 
    if return_eulers:
        return np.array([extract_eulers_from_rotation_matrix_XYZ(R_i) for R_i in R])    
    else:
        return R
    
def generate_rotation_matrix_from_eulers(eulers):
    """
    Given euler angles, generate rotation matrix using XYZ rotation
    Inputs:
        eulers: list of euler angles
    Output:
        np.ndarray, shape=(3,3) representing XYZ rotation
    """
    x, y, z = eulers[0], eulers[1], eulers[2]
    calc = np.array([
        [c(z), -s(z), 0.],
        [s(z), c(z), 0.],
        [0., 0., 1.] 
    ]) @ np.array([
        [c(y), 0., s(y)],
        [0., 1., 0.],
        [-s(y), 0., c(y)]
    ]) @ np.array([
        [1., 0., 0.],
        [0., c(x), -s(x)],
        [0., s(x), c(x)]
    ])
    assert np.allclose(calc, R.from_euler('xyz', eulers).as_matrix())
    return calc

def eight_point_algorithm(left_frame_pts,
                          right_frame_pts):
    """
        Given correspondences between two camera views, estimate the fundamental matrix
        Fundamental matrix contains the relative pose transform from left cam to right cam
        Inputs:
            left_frame_pts (np.ndarray): points in left frame. shape=(8, 2)
            right_frame_pts (np.ndarray): points in right frame. shape=(8, 2)
    """
    #TODO add an assert that the points are normalized
    u_l, u_r = left_frame_pts[:, 0], right_frame_pts[:, 0]
    v_l, v_r = left_frame_pts[:, 1], right_frame_pts[:, 1]
    A = np.stack([u_l*u_r, u_l*v_r, u_l, v_l*u_r, v_l*v_r, v_l, u_r, v_r, np.ones_like(u_l)], axis=1) # (N, 9)

    # construct initial F estimate
    U, S, Vh = np.linalg.svd(A)
    F_est = Vh[-1].reshape((3, 3))

    # Enforce rank-2 constraint
    U, S, Vh = np.linalg.svd(F_est)
    S[-1] = 0
    F = U @ np.diag(S) @ Vh

    return F

def DLT(pixels,
        projections):
    """
    Triangulate 3D point given 2D pixel coordinates using the Direct Linear Transform
    Inputs:
        pixels: np.ndarray - pixel coordinates for feature point. shape = N, 2 where N = num_cams
        projections: np.ndarray - projection matrices for each camera relative to the origin point. shape = (N,3,4) where N = num_cam
            *Note: first proj matrix should be identity as cam1 serves as the origin of the world coordinate frame
    Outputs:
        3-element np.ndarray representing best guess for triangulated 3D point  
    """
    # vectorized creation of A - kinda overkill for four cams, but good numpy practice lol
    N = pixels.shape[0]
    row1 = pixels[:, 1][:, np.newaxis] * projections[:, 2, :] - projections[:, 1, :] # vP_2 - P_1
    row2 = projections[:, 0, :] - pixels[:, 0][:, np.newaxis] * projections[:, 2, :] # P_0 - uP_2
    A = np.empty((2*N, 4), dtype=np.float64)
    A[0::2] = row1
    A[1::2] = row2
    U, S, Vh = np.linalg.svd(A)
    X = Vh[-1, :]
    X = X / X[-1]
    return X[:-1]

def ba_calc_residuals(x: np.ndarray,
                      n_cams: int,
                      obs_3d: np.ndarray,
                      obs_2d: np.ndarray):
    """
    Calculate residuals between projected 3D points and 2D observations to be optimized using bundle adjustment.
    Format of this function is to follow the specification of method 'fun' outlined by scipy.optimize.least_squares (i.e.
    fun(x0, *args, **kwargs))
    Inputs:
        x: np.ndarray - vector that will be optimized in the bundle adjustment process. should contain initial guess for camera projection matrices 
                        + estimates for 3D points as calculated by DLT. Shape = (n_cams * 12 + n_obs * 3)
        n_cams: int - number of cameras
        obs_2d: np.ndarray - matrix containing 2D pixel observations for each camera in each timestep. shape = (n_obs * n_cams, 2)
    Output:
        np.ndarray of shape (n,) where n is the number of total residuals = n_obs * n_cams * 2
        Note: the extra 2 is because there is a residual for each of the (u, v) in the observations
    """
    n_obs = obs_2d.shape[0] // n_cams
    Ps = x.reshape((n_cams, 3, 4))
    obs_3d = obs_3d.T
    # Ps = x[:n_cams*12].reshape((n_cams, 3, 4))
    # obs_3d = x[n_cams*12:].reshape(n_obs, 3).T                              # (3, n_obs)
    obs_3d = np.vstack((obs_3d, np.ones((1, obs_3d.shape[1]))))             # add homo coords, (4, n_obs)
    projected_2d = Ps @ obs_3d                                              # (n_cams, 3, 4) @ (4, n_obs) = (n_cams, 3, n_obs)
    projected_2d = np.transpose(projected_2d, (2, 0, 1)).reshape(-1, 3)     # (n_obs * n_cams, 3). ordering of (2, 0, 1) is important to preserve proper interleaving
    projected_2d = projected_2d / projected_2d[:, -1][:, np.newaxis]
    projected_2d = projected_2d[:, :2]                                      # get rid of homo coords, (n_obs * n_cams, 2)
    residuals = (projected_2d-obs_2d).flatten()
    assert residuals.shape == (n_obs * n_cams * 2,)
    return residuals

def five_point_algorithm(points):
    pass

def RANSAC(points,
           evalute_fn):
    pass
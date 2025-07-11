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
    return R.from_matrix(rot_mtrx).as_euler('XYZ')

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
    Given euler angles, generate rotation matrix using ZYX intrinsic rotation
    Inputs:
        eulers: list of euler angles
    Output:
        np.ndarray, shape=(3,3) representing ZYX rotation
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
    assert np.allclose(calc, R.from_euler('ZYX', eulers).as_matrix()), f"\n{calc}\n{R.from_euler('ZYX', eulers).as_matrix()}"
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
    print(pixels.shape)
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

def construct_intrinsics(fx, fy, ox, oy):
    return np.array(
        [
            [fx, 0., ox],
            [0., fy, oy],
            [0., 0., 1.]
        ]
    )

def project_3d_to_2d(Ps, obs_3d):
    """
    Given a set of projection matrices and 3D points, project them onto the 2D image planes
    Inputs:
        Ps: np.ndarray - projection matrices. shape = (N, 3, 4)
        obs_3d: np.ndarray - 3D points. shape = (N, 3)
    Outputs:
        2D points. shape = (N, 2)
    """
    obs_3d = obs_3d.T                                                       # (N, 3) -> (3, N)
    obs_3d = np.vstack((obs_3d, np.ones((1, obs_3d.shape[1]))))             # add homo coords, (4, n_eval)
    projected_2d = Ps @ obs_3d                                              # (n_cams, 3, 4) @ (4, n_eval) = (n_cams, 3, n_eval)
    projected_2d = np.transpose(projected_2d, (2, 0, 1)).reshape(-1, 3)     # (n_eval * n_cams, 3). ordering of (2, 0, 1) is important to preserve proper interleaving
    projected_2d = projected_2d / projected_2d[:, -1][:, np.newaxis]
    projected_2d = projected_2d[:, :2]                                      # get rid of homo coords, (n_eval * n_cams, 2)
    projected_2d = np.round(projected_2d)
    return projected_2d

def project_2d_to_3d(n_cams, Ps, obs_2d):
    """
    Given a set of projection matrices and 2D observation points for each camera, identify 3D points
    Inputs:
        n_cams: int - number of cams
        Ps: np.ndarray - projection matrices. shape = (N, 3, 4)
        obs_2d: np.ndarray - 2D image points. shape = (n_cams * N, 2). Each set of n_cams rows should represent the measurements of a single point
    Outputs:
        3D observation points. shape = (N, 3)
    """
    n_eval = obs_2d.shape[0] // n_cams
    obs_3d = np.empty((n_eval, 3), dtype=np.float32)
    for i in range(n_eval):
        pixels = obs_2d[n_cams*i:n_cams*(i+1), :]
        obs_3d[i] = DLT(pixels=pixels,
                        projections=Ps)
    return obs_3d

def homogenize_Ps(Ps):
    """
    Given a set of projection matrices, make them into homogeneous projection matrices
    Inputs:
        Ps: np.ndarray - projection matrices. shape = (N, 3, 4)
    Outputs:
        homogenous projection matrices. shape = (N, 4, 4)
    """
    if len(Ps.shape) == 2:
        Ps = Ps[np.newaxis, :, :]
    N = Ps.shape[0]
    Ps_homo = np.tile(np.eye(4), (N, 1, 1))
    for i in range(N):
        Ps_homo[i, :3, :3] = Ps[i, :3, :3]
        Ps_homo[i, :3, 3] = Ps[i, :3, 3]
    return Ps_homo.squeeze()

def compose_Ps(A, B):
    """
    Given two projection matrices, compose their transformations
    Inputs:
        A: np.ndarray - projection matrix. shape = (N, 3, 4)
        B: np.ndarray - projection matrix. shape = (N, 3, 4)
    Outputs:
        A @ B. composed transformation. shape = (N, 3, 4)
    """
    assert len(A.shape) == len(B.shape)
    A_homo = homogenize_Ps(A)
    B_homo = homogenize_Ps(B)
    if len(A_homo.shape) == 2:
        return (A_homo @ B_homo)[:3, :]
    else:
        return (A_homo @ B_homo)[:, :3, :]

def construct_extrinsics(pos,
                         eulers):
    """
    Construct extrinsic matrix given an input translation and eulers
    Inputs:
        pos: np.ndarray - translation vector. shape = (3, 1)
        eulers: np.ndarray - eulers to be applied in XYZ. shape = (3, 1)
    Output:
        ext_wc - world to camera frame
        ext_cw - camera to world frame
    """
    R_cw = generate_rotation_matrix_from_eulers(eulers) # body -> world, +x forward, +y left, +z up
    R_wc = R_cw.T # world -> body
    # need to transform to +x left, +y up, +z forward
    # this ensures optical axis is aligned with +z enabling proper homogenous calculation
    # R_wc = np.array([
    #     [0., 1., 0.,], 
    #     [0., 0., 1.], 
    #     [1., 0., 0.,]
    # ]) @ R_wc
    R_wc = np.array([
        [0., 0., 1.,], 
        [1., 0., 0.], 
        [0., 1., 0.,]
    ]) @ R_wc
    t = -R_wc @ pos
    ext_wc = np.hstack((R_wc, t))
    
    # can't directly use R_cw here because R_wc includes a camera frame transform. 
    # Therefore, to be completel correct, best to just do R_wc.T which includes that frame transform
    ext_cw = np.hstack((R_wc.T, pos))
    return ext_wc, ext_cw

def ba_calc_residuals(x: np.ndarray,
                      n_cams: int,
                      intrinsics: np.ndarray,
                      obs_2d: np.ndarray):
    """
    Calculate residuals between projected 3D points and 2D observations to be optimized using bundle adjustment.
    Format of this function is to follow the specification of method 'fun' outlined by scipy.optimize.least_squares (i.e.
    fun(x0, *args, **kwargs))
    Inputs:
        x: np.ndarray - vector that will be optimized in the bundle adjustment process. should contain initial guess for camera translation and eulers 
                        + estimates for 3D points as calculated by DLT. shape = (n_cams * 6 + n_obs * 3)
        n_cams: int - number of cameras
        intrinsics: np.ndarray - intrinsic matrix for each camera. shape = (n_cams, 3, 3)
        obs_2d: np.ndarray - matrix containing 2D pixel observations for each camera in each timestep. shape = (n_obs * n_cams, 2)
    Output:
        np.ndarray of shape (n,) where n is the number of total residuals = n_obs * n_cams * 2
        Note: the extra 2 is because there is a residual for each of the (u, v) in the observations
    """
    n_obs = obs_2d.shape[0] // n_cams

    # construct projection matrices
    Ps = np.empty((n_cams, 3, 4))
    Ps[0] = intrinsics[0] @ np.hstack((np.eye(3), np.zeros((3, 1))))
    for i in range(n_cams-1):
        params = x[6*i:6*(i+1)]
        pos = params[:3].reshape((3, 1))
        rot_vec = params[3:6].reshape((3,))
        rot_mtrx = R.from_rotvec(rot_vec).as_matrix()
        ext_wc = np.hstack((rot_mtrx, pos))
        # ext_wc, _ = construct_extrinsics(pos, eulers)
        intrinsic = intrinsics[i+1, :, :] 
        Ps[i+1, :, :] = intrinsic @ ext_wc

    # # estimate 3d points given current estimate of projection matrices 
    # obs_3d = x[n_cams*6:].reshape((n_obs, 3))

    obs_3d = np.empty((n_obs, 3), dtype=np.float32)
    for i in range(n_obs):
        obs_3d[i] = DLT(pixels=obs_2d[n_cams*i:n_cams*(i+1)],
                        projections=Ps)
        # breakpoint()
    
    # calculate residuals
    obs_3d = obs_3d.T                                                       # (3, n_obs)
    obs_3d = np.vstack((obs_3d, np.ones((1, obs_3d.shape[1]))))             # add homo coords, (4, n_obs)
    projected_2d = Ps @ obs_3d                                              # (n_cams, 3, 4) @ (4, n_obs) = (n_cams, 3, n_obs)
    # breakpoint()
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
import numpy as np
from numpy import cos as c
from numpy import sin as s
from scipy.spatial.transform import Rotation as R

def extract_eulers_from_rotation_matrix_XYZ(R):
    """
    Given a rotation matrix R, extract eulers assuming XYZ rotation
    Inputs:
        R: np.ndarray - rotation matrix
    """
    theta_y = np.arcsin(-R[2, 0])
    theta_x = np.arctan2(R[2, 1] / np.cos(theta_y), R[2, 2] / np.cos(theta_y))
    theta_z = np.arctan2(R[1, 0] / np.cos(theta_y), R[0, 0] / np.cos(theta_y))
    # theta_x = np.arctan2(R[2, 1], R[2, 2])
    # theta_y = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    # theta_z = np.arctan2(R[1, 0], R[0, 0])
    return np.array([theta_x, theta_y, theta_z])

def lookat(origin, target, up, return_eulers=True):
    """
    Given a target pos, calculate the rotation needed to orient the body in the direction of the target
    Inputs:
        origin: np.ndarray - (N, 3) matrix indicating cam positions
        target: np.ndarray - (N, 3) matrix indicating target positions
        up: np.ndarray - (1, 3) indicating world up (i.e. +Z)
        return_eulers: bool - if True, extract eulers from rotation matrix and return those. Otherwise, return rotation matrix
            *Note: eulers are calculated assuming a XYZ rotation
    """
    assert np.allclose(np.linalg.norm(up), 1.0), "up vector is not normalized"
    forward = (target-origin) / np.linalg.norm(target-origin, axis=1, keepdims=True)
    left = np.cross(up, forward)
    left = left / np.linalg.norm(left, keepdims=True)
    body_up = np.cross(forward, left)
    body_up = body_up / np.linalg.norm(body_up, keepdims=True)
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

def five_point_algorithm(points):
    pass

def RANSAC(points,
           evalute_fn):
    pass
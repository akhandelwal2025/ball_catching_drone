import numpy as np
from numpy import cos as c
from numpy import sin as s

def generate_rotation_matrix_from_eulers(eulers):
    """
    Given euler angles, generate rotation matrix using XYZ rotation
    Inputs:
        eulers: list of euler angles
    Output:
        np.ndarray, shape=(3,3) representing XYZ rotation
    """
    x, y, z = eulers[0], eulers[1], eulers[2]
    return np.array([
        [1., 0., 0.],
        [0., c(x), -s(x)],
        [0., s(x), c(x)]
    ]) @ np.array([
        [c(y), 0., s(y)],
        [0., 1., 0.],
        [-s(y), 0., c(y)]
    ]) @ np.array([
        [c(z), -s(z), 0.,],
        [s(z), c(z), 0.],
        [0., 0., 1.] 
    ])

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
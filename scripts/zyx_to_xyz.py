from scipy.spatial.transform import Rotation as R
import numpy as np

np.set_printoptions(precision=3, suppress=True)

# cam eulers - to be applied ZYX
cam1_zyx = np.radians(np.array([0., 25., 45.,], dtype=np.float32))
cam2_zyx = np.radians(np.array([0., 25., 315.,], dtype=np.float32))
cam3_zyx = np.radians(np.array([0., 25., 225.,], dtype=np.float32))
cam4_zyx = np.radians(np.array([0., 25., 135.,], dtype=np.float32))

# transform from ZYX to XYZ
cam1_R = R.from_euler("ZYX", cam1_zyx, degrees=False).as_matrix()
cam1_xyz = R.from_matrix(cam1_R).as_euler("XYZ", degrees=False)

cam2_R = R.from_euler("ZYX", cam2_zyx, degrees=False).as_matrix()
cam2_xyz = R.from_matrix(cam2_R).as_euler("XYZ", degrees=False)

cam3_R = R.from_euler("ZYX", cam3_zyx, degrees=False).as_matrix()
cam3_xyz = R.from_matrix(cam3_R).as_euler("XYZ", degrees=False)

cam4_R = R.from_euler("ZYX", cam4_zyx, degrees=False).as_matrix()
cam4_xyz = R.from_matrix(cam4_R).as_euler("XYZ", degrees=False)

# print out results
print(f"cam1_zyx: {cam1_zyx} -> cam1_xyz: {cam1_xyz}")
print(f"cam2_zyx: {cam2_zyx} -> cam2_xyz: {cam2_xyz}")
print(f"cam3_zyx: {cam3_zyx} -> cam3_xyz: {cam3_xyz}")
print(f"cam4_zyx: {cam4_zyx} -> cam4_xyz: {cam4_xyz}")

print(R.from_euler("XYZ", cam1_zyx).as_matrix())
print(R.from_euler("ZYX", cam1_zyx).as_matrix())
print(np.allclose(R.from_euler("XYZ", cam1_xyz).as_matrix(), R.from_euler("ZYX", cam1_zyx).as_matrix()))
print(np.allclose(R.from_euler("XYZ", cam2_xyz).as_matrix(), R.from_euler("ZYX", cam2_zyx).as_matrix()))
print(np.allclose(R.from_euler("XYZ", cam3_xyz).as_matrix(), R.from_euler("ZYX", cam3_zyx).as_matrix()))
print(np.allclose(R.from_euler("XYZ", cam4_xyz).as_matrix(), R.from_euler("ZYX", cam4_zyx).as_matrix()))

def R_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0.],
        [np.sin(theta), np.cos(theta), 0.],
        [0., 0., 1.]
    ])

def R_y(theta):
    return np.array([
        [np.cos(theta), 0., np.sin(theta)],
        [0., 1., 0.,],
        [-np.sin(theta), 0., np.cos(theta)]
    ])

print(R_y(0.436) @ R_z(0.785))

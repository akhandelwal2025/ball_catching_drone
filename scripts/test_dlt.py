from src.utils import DLT
import numpy as np


arrs = np.load('data/fake_imgs_10.npz')
fake_imgs = arrs['fake_imgs']
pts_2d = arrs['pts_2d']
pts_3d = arrs['pts_3d']
projections = arrs['projections']
extrinsics_cw = arrs['extrinsics_cw']
N = fake_imgs.shape[0]
avg_dist_error = 0.0
for i in range(N):
    pixels = pts_2d[i]
    pt_3d = pts_3d[i]
    est_pt_3d_body = DLT(pixels, projections)
    dist_error = np.linalg.norm(pt_3d - est_pt_3d_body)
    print(f'dist_error: {dist_error}')
    avg_dist_error += dist_error
print(f"avg_dist_error: {avg_dist_error/N}")
# -------------- OLD -------------- 
# pixel1 = [160, 120]
# pixel2 = [80, 100]
# proj1 = np.array([
#     [1., 0., 0., 0.,],
#     [0., 1., 0., 0.,],
#     [0., 0., 1., 0.,]
# ])
# proj2 = np.array([
#     [0.5, -0.5,  0.707, 1.],
#     [0.854, 0.146, -0.5, 1.],
#     [0.146, 0.854, 0.5, 1. ]
# ]) # represents 45 deg rotation in all axes applied XYZ + translation of [1., 1., 1.]
# pixels = np.stack((pixel1, pixel2), axis=0)
# projs = np.stack((proj1, proj2), axis=0)
# pt = DLT(pixels, projs)
# print(pt)

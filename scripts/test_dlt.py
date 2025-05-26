from src.utils import DLT
import numpy as np

pixel1 = [160, 120]
pixel2 = [80, 100]
proj1 = np.array([
    [1., 0., 0., 0.,],
    [0., 1., 0., 0.,],
    [0., 0., 1., 0.,]
])
proj2 = np.array([
    [0.5, -0.5,  0.707, 1.],
    [0.854, 0.146, -0.5, 1.],
    [0.146, 0.854, 0.5, 1. ]
]) # represents 45 deg rotation in all axes applied XYZ + translation of [1., 1., 1.]
pixels = np.stack((pixel1, pixel2), axis=0)
projs = np.stack((proj1, proj2), axis=0)
pt = DLT(pixels, projs)

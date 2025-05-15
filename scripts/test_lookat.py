import src.utils as utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def draw_axes(ax,
              origin: np.ndarray,
              rotation: np.ndarray = None,
              eulers: np.ndarray = None):
    N = origin.shape[0]
    if rotation is None:
        rotation = np.empty((N, 3, 3))
        rotation = np.apply_along_axis(func1d=utils.generate_rotation_matrix_from_eulers, 
                                        axis=1,
                                        arr=eulers)
        print(f"computed rotation: {rotation}")
    for i in range(N):
        ax.quiver(origin[i, 0], origin[i, 1], origin[i, 2], *rotation[i, :, 0], color='red', pivot='tail', length=2, normalize=True)
        ax.quiver(origin[i, 0], origin[i, 1], origin[i, 2], *rotation[i, :, 1], color='green', pivot='tail', length=2, normalize=True)
        ax.quiver(origin[i, 0], origin[i, 1], origin[i, 2], *rotation[i, :, 2], color='blue', pivot='tail', length=2, normalize=True)

origin = np.array([0., 0., 0.])[np.newaxis, :]
target = np.array([1., -1., -1.])[np.newaxis, :]
up = np.array([0., 0., 1.])[np.newaxis, :]
R = utils.lookat(origin, target, up, return_eulers=False)
eulers = utils.lookat(origin, target, up, return_eulers=True)
print(f"rotation from lookat: {R}, determinant: {np.linalg.det(R)}")
print(f"eulers from lookat: {eulers}")
# print(eulers)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-2, 10)  # Set x-axis limits to a range around zero
ax.set_ylim(-2, 10)  # Set y-axis limits to a range around zero
ax.set_zlim(0, 12)

# draw_axes(ax, origin, rotation=np.eye(3)[np.newaxis, :, :])
# draw_axes(ax, origin, rotation=R)

draw_axes(ax, origin, eulers=np.array([0., 0., 0.,])[np.newaxis, :])
draw_axes(ax, origin, eulers=eulers)
plt.show()

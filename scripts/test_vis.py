from src.vis import Vis
import yaml
import numpy as np

with open('cfgs/PSEyeMocap.yaml', "r") as file:
    cfg = yaml.safe_load(file)

cam_pos = np.stack([cfg['pos']['cam1'],
                    cfg['pos']['cam2'],
                    cfg['pos']['cam3'],
                    cfg['pos']['cam4']], axis=0)
cam_eulers = np.stack([cfg['eulers']['cam1'],
                       cfg['eulers']['cam2'],
                       cfg['eulers']['cam3'],
                       cfg['eulers']['cam4']], axis=0)
cam_eulers = np.radians(cam_eulers)

vis = Vis(cfg, cam_pos, cam_eulers)
while True:
    vis.render()


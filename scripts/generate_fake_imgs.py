import numpy as np
import argparse
from src.env import TestEnv
import yaml

DEFAULT_NUM_TIMESTEPS = 10

def main(args):
    with open(args.env_cfg, "r") as file:
        cfg = yaml.safe_load(file)
    env = TestEnv(cfg)
    w, h = cfg['img_resolution'][0], cfg['img_resolution'][1] 
    fake_imgs = np.empty((args.num_timesteps, 4, h, w)) # n_timesteps, n_cams, height, width
    pts_2d = np.empty((args.num_timesteps, 4, 2))
    pts_3d = np.empty((args.num_timesteps, 3))
    projections = env.projections
    extrinsics_cw = env.extrinsics_cw
    i = 0
    while i < args.num_timesteps:
        feature_pt = env.gen_random_pt()
        if env.pt_in_fovs(feature_pt) and env.gen_imgs():
            env.render()
            fake_imgs[i, ...] = env.fake_imgs
            pts_2d[i, :, :] = env.projected_pts
            pts_3d[i, :] = env.feature_pt
            i += 1
    np.savez(args.save_filename, fake_imgs=fake_imgs, pts_2d=pts_2d, pts_3d=pts_3d, projections=projections, extrinsics_cw=extrinsics_cw)

if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--env_cfg', type=str, default='cfgs/test_env.yaml')
    parser.add_argument('--num_timesteps', type=int, default=DEFAULT_NUM_TIMESTEPS)
    parser.add_argument('--save_filename', type=str, default=f'data/fake_imgs_{DEFAULT_NUM_TIMESTEPS}.npz')
    args = parser.parse_args()  
    main(args)
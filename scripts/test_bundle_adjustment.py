import numpy as np
import argparse
from src.env import TestEnv
from src.mocap import Mocap
import src.utils as utils 
import yaml

def main(args):
    with open(args.mocap_cfg, "r") as file:
        mocap_cfg = yaml.safe_load(file)
    mocap = Mocap(mocap_cfg)
    n_cams = mocap.n_cams
    projections = mocap.projections
    obs_2d = np.empty((n_cams * args.n_obs, 2), dtype=int)
    obs_3d = np.empty((args.n_obs, 3), dtype=np.float32)
    for i in range(args.n_obs):
        imgs, timesteps = mocap.read_cameras()
        pixels = mocap.locate_centers(imgs, 
                                       num_centers=1, 
                                       lower=200,
                                       upper=256).squeeze() 
        obs_3d[i] = utils.DLT(pixels, projections)
        obs_2d[n_cams*i:n_cams*(i+1)] = pixels
    x0 = np.concatenate((projections.flatten(), 
                         obs_3d.flatten()))
    residuals = utils.ba_calc_residuals(x0, n_cams, obs_2d)
    breakpoint()
        
if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--mocap_cfg', type=str, default='cfgs/mocap.yaml')
    parser.add_argument('--n_obs', type=int, default=5)
    args = parser.parse_args()  
    main(args)
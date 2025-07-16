from pseyepy import Camera
from src.mocap import PsEyeMocap
import time
import cv2
import numpy as np
import argparse
import yaml
import src.utils as utils

# BLACK HSV BOUNDS
LOWER = np.array([50, 50, 50], dtype=np.uint8)
UPPER = np.array([255, 255, 255], dtype=np.uint8)
N_CAMS = 4

def main(args):
    with open(args.mocap_cfg, "r") as file:
        cfg = yaml.safe_load(file)
    mocap = PsEyeMocap(cfg)
    ext_c1w = mocap.extrinsics_cw[0]
    ext_wc1 = mocap.extrinsics_wc[0]
    for i in range(N_CAMS):
        print(f"--------- testing cam {i} ---------")
        ext_wc = mocap.extrinsics_wc[i]
        ext_cw = mocap.extrinsics_cw[i]
        ext_c1c = mocap.extrinsics_c1c[i]
        ext_cc1 = mocap.extrinsics_cc1[i]
        pos_wf = mocap.cam_pos[i][np.newaxis, :]

        # test that wc, cw are inverse operations - multiplying them should result in identity rotation and zero translation
        assert np.allclose(utils.compose_Ps(ext_wc, ext_cw), np.array([
            [1., 0., 0., 0.,],
            [0., 1., 0., 0.,],
            [0., 0., 1., 0.,]
        ]))
        print("verified wc, cw inverse")

        # verify arbitrary roundtrip from world -> cam -> world
        pt_wf = np.random.uniform(0, 3, size=(10, 3))
        pt_cf = utils.transform(ext_wc, pt_wf)
        pt_wf_back = utils.transform(ext_cw, pt_cf)
        assert np.allclose(pt_wf, pt_wf_back)
        print("verified round-trip world -> cam -> world")

        # test that c1 -> c is just c1 -> w -> c
        assert np.allclose(utils.compose_Ps(ext_wc, ext_c1w), ext_c1c)
        print("verified c1c")

        # test that c -> c1 is just c -> w -> c1
        assert np.allclose(utils.compose_Ps(ext_wc1, ext_cw), ext_cc1)
        print("verified cc1")

        # test that c1c and cc1 are inverse operations
        assert np.allclose(utils.compose_Ps(ext_cc1, ext_c1c), np.array([
            [1., 0., 0., 0.,],
            [0., 1., 0., 0.,],
            [0., 0., 1., 0.,]
        ]))
        print("verified c1c, cc1 inverse")

        # test that wf cam position gets transformed to 0, 0, 0 in cf
        assert np.allclose(utils.transform(ext_wc, pos_wf), np.array([0., 0., 0.]))
        print("verified wf cam pos gets transformed to 0, 0, 0")

        # test that 0, 0, 0 in cf gets transformed to proper wf cam position
        origin = np.array([0., 0., 0.,]).reshape((1, 3))
        assert np.allclose(utils.transform(ext_cw, origin), pos_wf)
        print("verified origin -> wf == wf cam pos")

        # Extract R, t from ext_cw (cam -> world)
        R = ext_cw[:, :3]
        t = ext_cw[:, 3]

        cam_pos_from_ext = (R @ np.array([0., 0., 0.]) + t).T  # R @ origin + t
        assert np.allclose(cam_pos_from_ext, pos_wf), f"pos_wf: {pos_wf} | cam_pos_from_ext: {cam_pos_from_ext}"
        print("verified cam position extracted from ext_cw matches given pos_wf")
    breakpoint()


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--mocap_cfg', type=str, default='cfgs/PSEyeMocap.yaml')
    args = parser.parse_args()  
    main(args)
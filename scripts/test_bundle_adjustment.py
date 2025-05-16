import argparse
from src.env import TestEnv
import yaml

def main(args):
    with open(args.env_cfg, "r") as file:
        cfg = yaml.safe_load(file)
    env = TestEnv(cfg)
    while True:
        feature_pt = env.gen_random_pt()
        print(env.pt_in_fovs(feature_pt))
        env.render()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--env_cfg', type=str, default='cfgs/test_env.yaml')
    args = parser.parse_args()  
    main(args)
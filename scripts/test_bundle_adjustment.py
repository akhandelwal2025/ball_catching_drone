import argparse
from src.env import TestEnv
import yaml

def main(args):
    with open(args.env_cfg, "r") as file:
        cfg = yaml.safe_load(file)
    env = TestEnv(cfg)
    env.render()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--env_cfg', type=str, default='cfgs/test_env.yaml')
    args = parser.parse_args()  
    main(args)
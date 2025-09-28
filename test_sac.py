from sac.utils import str2bool, evaluate_policy, Action_adapter, Action_adapter_reverse, Reward_adapter
from datetime import datetime
from sac.SAC import SAC_countinuous
import gymnasium as gym
import os, shutil
import argparse
import torch
import random
from Env import Env
import numpy as np

import time

# 新增：tensorboard相关导入
# from torch.utils.tensorboard import SummaryWriter

# 风险预测模型导入
# from algos.transformer_risk_model_with_B import TransformerRiskPredictor
from algos.transformer_risk_model import TransformerRiskPredictor

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(5e5), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(1e4), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(2.5e3), help='Model evaluating interval, in steps.')
parser.add_argument('--update_every', type=int, default=50, help='Training Fraquency, in stpes')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=3e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=3e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size of training')
parser.add_argument('--alpha', type=float, default=0.12, help='Entropy coefficient')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive_alpha or Not')

parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--max_e_steps', type=int, default=1000)

parser.add_argument('--test_logs_path', type="str", help='Test save path')

opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)

def main():
    env_name = "unsignalizedIntersections"

    env = Env({
        "junction_list": ['229', '499', '332', '334'],  #
        "spawn_rl_prob": {},
        "probablity_RL": 1.0,
        "cfg": 'real_data/osm.sumocfg',  #
        "render": False,  # 可视化
        "map_xml": 'real_data/CSeditClean_1.net_threelegs.xml',
        "max_episode_steps": opt.max_e_steps,
        "conflict_mechanism": 'flexible',
        "traffic_light_program": {
            "disable_state": 'G',
            "disable_light_start": 0
        },
        # 风险预测模型
        "use_safety_predictor": False,
    })


    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    # load model
    agent = SAC_countinuous(**vars(opt))

    run_num = 36
    timestep = 500
    agent.load(env_name,timestep=timestep, run_num=run_num)

    state, _ = env.reset()
    for _ in range(1, opt.max_e_steps + 1):
        while not state:
            state, _, _, _, _, _ = env.step()
        
        s = list(state.values())
        a = agent.select_action(s, deterministic=True)
        a = a.flatten()
        action = a * opt.max_action
        action_dict = dict(zip(state.keys(), action))



        state, rewards, is_terminal, _, _, _ = env.step(action_dict)
        for key, done in is_terminal.items():
            if done:
                state.pop(key)

    env.monitor.evaluate()
    env.monitor.save_to_pickle(file_name=opt.test_logs_path)

    env.close()


if __name__ == '__main__':
    main()

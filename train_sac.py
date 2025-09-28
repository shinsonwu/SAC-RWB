from sac.utils import str2bool, evaluate_policy, Action_adapter, Action_adapter_reverse, Reward_adapter
from datetime import datetime
from sac.SAC import SAC_countinuous
import os, shutil
import argparse
import torch
from Env import Env
import numpy as np

import time

from torch.utils.tensorboard import SummaryWriter

from algos.transformer_risk_model import TransformerRiskPredictor

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
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

parser.add_argument('--d_model', type=int, default=128, help='Feature dimension of the transformer model (d_model)')
parser.add_argument('--nhead', type=int, default=4, help='Number of heads in the multi-head attention mechanism')
parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the transformer encoder')
parser.add_argument('--buffer_size', type=int, default=10000, help='Buffer size for the risk prediction model')
parser.add_argument('--risk_model_path', type=str, default=10000, help='Buffer size for the risk prediction model')
parser.add_argument('--load_risk_model', type=int, default=0, help='Whether to load the pretrained risk prediction model (1 to load, 0 not to load)')

parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--max_e_steps', type=int, default=1000)
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)

def main():
    env_name = "unsignalizedIntersections"

    # 创建风险预测模型保存目录
    risk_model_dir = "risk_model/"
    if not os.path.exists(risk_model_dir):
        os.makedirs(risk_model_dir)

    # 创建风险预测模型
    risk_predictor = TransformerRiskPredictor(
        state_dim=96+1+1,
        action_dim=1,
        d_model=opt.d_model,
        nhead=opt.nhead,
        num_layers=opt.num_layers,
        buffer_size=opt.buffer_size
    ).to(opt.dvc)

    if opt.load_risk_model:
        risk_model_path = opt.risk_model_path
        if os.path.exists(risk_model_path):
            try:
                risk_predictor.load_model(risk_model_path)
            except Exception as e:
                print(f"error: {e}")
        else:
            print(f"Risk prediction model file not found: {risk_model_path}")
    else:
        print("Not loading pre trained risk prediction models, training will start from scratch")


    # Build Env
    env = Env({
        "junction_list": ['229', '499', '332', '334'],  #
        "spawn_rl_prob": {},  #
        "probablity_RL": 1.0,  #
        "cfg": 'real_data/osm.sumocfg',  #
        "render": opt.render,  # 可视化
        "map_xml": 'real_data/CSeditClean_1.net_threelegs.xml',
        "max_episode_steps": opt.max_e_steps,
        "conflict_mechanism": 'flexible',
        "traffic_light_program": {
            "disable_state": 'G',
            "disable_light_start": 0
        },
        "use_safety_predictor": True,
        "risk_predictor": risk_predictor
    })

    opt.state_dim = env.n_obs
    opt.action_dim = env.n_obs
    opt.max_action = env.max_acc   #remark: action space【-max,max】

    # Seed Everything
    # env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    # Build DRL model
    if not os.path.exists('model'): os.mkdir('model')
    agent = SAC_countinuous(**vars(opt)) # var: transfer argparse to dictionary

    ###### log ######
    log_dir = "Train_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    log_f_name = log_dir + 'SAC_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)

    tensorboard_dir = "tensorboard_logs"
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    tensorboard_dir = tensorboard_dir + '/' + env_name + '/'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    
    n = len(next(os.walk(tensorboard_dir))[1])
    # 创建tensorboard writer
    timenow = str(datetime.now())[0:-10]
    timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
    writepath = tensorboard_dir + '/run{}'.format(run_num) + timenow
    writer = SummaryWriter(writepath)
    print("tensorboard logging at : " + tensorboard_dir + f'run_{run_num}')

    total_steps = 0
    cur_time = time.localtime()

    print("start time: %02d:%02d:%02d" % (cur_time.tm_hour, cur_time.tm_min, cur_time.tm_sec))
    for i_episode in range(opt.epochs):
        state, _ = env.reset()
        stime = time.time()

        current_ep_reward = 0
        current_ep_efficiency_reward = 0
        current_ep_safety_reward = 0
        current_reward_count = 0

        '''Interact & trian'''
        for t in range(opt.max_e_steps):
            total_steps += 1
            while not state:
                state, _, _, _, _, _ = env.step()
            s = list(state.values())
            a = agent.select_action(s, deterministic=False)  # a∈[-1,1]
            a = a.flatten()
            action = a * opt.max_action

            safe_reward_dict = {}
            state_batch_for_risk = []
            action_batch_for_risk = []
            key_list = []
            action_dict = dict(zip(state.keys(), action))
            a_dict = dict(zip(state.keys(), a))

            for key, state_val in state.items():
                veh_id = env.convert_virtual_id_to_real_id(key)
                action_val = np.array([action_dict[key]])

                risk_predictor.buffer.add(veh_id, state_val, action_val)

            if env.use_safety_predictor:
                for key in state.keys():
                    veh_id = env.convert_virtual_id_to_real_id(key)
                    s_seq, a_seq = risk_predictor.buffer.get_sequence(veh_id)

                    if s_seq is not None:
                        state_batch_for_risk.append(s_seq)
                        action_batch_for_risk.append(a_seq)
                        key_list.append(key)

            if state_batch_for_risk:
                s_tensor = torch.tensor(np.concatenate(state_batch_for_risk), dtype=torch.float32).to(opt.dvc)
                a_tensor = torch.tensor(np.concatenate(action_batch_for_risk), dtype=torch.float32).to(opt.dvc)

                with torch.no_grad():
                    risk_scores = risk_predictor(s_tensor, a_tensor).cpu().numpy().flatten()

                    for idx, key in enumerate(key_list):
                        safe_reward_dict[key] = risk_scores[idx] # risk_score 在 [0, 1] 之间

            next_state, reward, is_terminal, truncated, info, new_state = env.step(action_dict, safe_reward_dict)

            for i, veh_id in enumerate(reward.keys()):
                agent.replay_buffer.add(state[veh_id], np.array(a_dict[veh_id]), reward[veh_id], new_state[veh_id], True)

            for key, done in is_terminal.items():
                if done:
                    next_state.pop(key)
            state = next_state.copy()

            # 统计各类奖励
            for veh_id, reward_val in reward.items():
                current_ep_reward += reward_val
                current_reward_count += 1
                if veh_id in env.reward_record:
                    current_ep_efficiency_reward += env.reward_record[veh_id]['efficiency']
                    current_ep_safety_reward += env.reward_record[veh_id]['safety']


            '''train if it's time'''
            if total_steps % (0.2*opt.max_e_steps) == 0:
                losses = agent.train()
                writer.add_scalar('Loss/q_loss', losses['q_loss'], total_steps)
                writer.add_scalar('Loss/a_loss', losses['a_loss'], total_steps)
                writer.add_scalar('Loss/alpha_loss', losses['alpha_loss'], total_steps)

                if env.use_safety_predictor and len(risk_predictor.buffer.active_buffer) >= 64:
                    loss = risk_predictor.train_model(batch_size=128, epochs=3, lr=1e-5, device=opt.dvc)
                    writer.add_scalar('Loss/Risk_Model_Loss', loss, total_steps)


            '''save model'''
            if total_steps % opt.save_interval == 0:
                print("save model")
                agent.save(env_name, int(total_steps/opt.steps), run_num)
                run_num_pretrained = run_num
                random_seed = 0
                risk_model_dir_path = risk_model_dir+"risk_{}_{}_{}_{}.pth".format(env_name, random_seed, i_episode, run_num_pretrained)
                risk_predictor.save_model(risk_model_dir_path)

        etime = time.time()
        print(f"\n{'=' * 100}\n"
              f"Episode: {i_episode:<10} | Timestep: {total_steps:<10} | Average Reward: {current_ep_reward:<10.2f} | Episode Time: {round((etime - stime), 2):<10.2f}\n"
              f"{'=' * 100}")

        # 记录奖励指标
        writer.add_scalar('Reward/Total', current_ep_reward, i_episode)
        writer.add_scalar('Reward/Efficiency', current_ep_efficiency_reward, i_episode)
        writer.add_scalar('Reward/Safety', current_ep_safety_reward, i_episode)

        # 计算平均奖励
        writer.add_scalar('Average_Reward/Avg_Total', current_ep_reward / current_reward_count, i_episode)
        writer.add_scalar('Average_Reward/Avg_Efficiency', current_ep_efficiency_reward / current_reward_count,
                          i_episode)
        writer.add_scalar('Average_Reward/Avg_Safety', current_ep_safety_reward / current_reward_count,
                          i_episode)

        # 记录环境数据
        env_data = env.monitor.get_data(env)
        writer.add_scalar('Env_Data/max_quene_length', env_data['max_quene_length'], total_steps)
        writer.add_scalar('Env_Data/avg_quene_length', env_data['avg_quene_length'], total_steps)
        writer.add_scalar('Env_Data/quene_wait', env_data['quene_wait'], total_steps)
        writer.add_scalar('Env_Data/conflict', env_data['conflict'], total_steps)
        writer.add_scalar('Env_Data/conflict_rate', env_data['conflict_rate'], total_steps)
        writer.add_scalar('Env_Data/avg_speed', env_data['avg_speed'], total_steps)

    env.close()


if __name__ == '__main__':
    main()

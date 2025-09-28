from typing import Set
from ray.rllib.utils.typing import AgentID
from core.sumo_interface import SUMO

from core.costomized_data_structures import Vehicle, Container
from core.NetMap import NetMap
from core.utils import rank_of_first
import numpy as np
import gym
import random, math
from gym.spaces.box import Box

from copy import deepcopy
from core.utils import start_edges, end_edges, dict_tolist, UNCONFLICT_SET
from gym.spaces import Discrete
from core.monitor import DataMonitor

WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
RED = (255, 0, 0)

BLACK = (97,0,255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

EPSILON = 0.00001


class Env(gym.Env):
    def __init__(self, config) -> None:
        ## TODO: use config to pass parameters

        super(Env, self).__init__()
        self.observation_space = Box(
            low=-1,
            high=1,
            shape=(self.n_obs,),
            dtype=np.float32)
        self.action_space = Discrete(2)

        self.config = config
        self.print_debug = False
        self.cfg = config['cfg']
        self.map_xml = config['map_xml']
        self.keywords_order = ['topstraight', 'topleft', 'rightstraight', 'rightleft', 'bottomstraight', 'bottomleft',
                               'leftstraight', 'leftleft']
        self._max_episode_steps = 1000  # unlimited simulation horizon
        if 'max_episode_steps' in config.keys():
            self._max_episode_steps = self.config['max_episode_steps']
        self.traffic_light_program = self.config['traffic_light_program']

        self.junction_list = self.config['junction_list']
        self.sumo_interface = SUMO(self.cfg, render=self.config['render'])
        self.map = NetMap(self.map_xml, self.junction_list)

        self.spawn_rl_prob = config['spawn_rl_prob']
        self.default_rl_prob = config['probablity_RL']  # 1.0
        self.rl_prob_list = config['rl_prob_range'] if 'rl_prob_range' in config.keys() else None

        self.start_edges = start_edges
        self.end_edges = end_edges

        self.max_acc = 3.0
        self.min_acc = -3.0
        self.max_speed = 10
        self.min_speed = 0

        self.control_distance = 30
        self.control_zone_length = 30
        self.max_wait_time = 200
        self.vehicle_len = 5.0

        self.init_env()
        self.previous_global_waiting = dict()
        self.global_obs = dict()

        for JuncID in self.junction_list:
            self.previous_global_waiting[JuncID] = dict()
            self.global_obs[JuncID] = 0
            for keyword in self.keywords_order:
                self.previous_global_waiting[JuncID][keyword] = 0
                self.previous_global_waiting[JuncID]['sum'] = 0

        ## off, standard, flexible
        if 'conflict_mechanism' in config:
            self.conflict_resolve_mechanism_type = config['conflict_mechanism']
        else:
            self.conflict_resolve_mechanism_type = 'off'
        

        # 风险预测模型
        self.use_safety_predictor = config['use_safety_predictor']
        if self.use_safety_predictor:
            self.risk_predictor = config['risk_predictor']

        self.use_deadlock_resolver = True  # 新增一个开关，方便启用/禁用此功能
        self.deadlock_thresholds = {
            'queue_len': 4,  # 触发检测的最小队列长度
            'wait_time': 25.0,  # 触发检测的队首车最小等待时间 (秒)
            'speed': 0.5,  # 队首车被视为“停滞”的最大速度 (m/s)
            'persistence': 3  # 死锁条件需要持续多少个step才被激活
        }

    @property
    def n_obs(self):
        ## TODO defination of obs
        return 80 + 16 + 1 + 1

    @property
    def n_action(self):
        ## TODO defination of ation
        return 1

    @property
    def env_step(self):
        return self._step

    def _print_debug(self, fun_str):
        if self.print_debug:
            print('exec: ' + fun_str + ' at time step: ' + str(self._step))


    def init_env(self):
        self.episode_total_vehicles = 0  # 记录 episode 中出现的总车辆数
        self.episode_collided_vehicles = 0  # 记录 episode 中发生碰撞的车辆数
        self.episode_arrived_vehicles = 0  # 记录 episode 中成功到达的车辆数
        self.episode_total_waiting_time = 0.0  # 记录所有到达车辆的总等待时间

        ## vehicle level
        self.vehicles = Container()  # 
        self.rl_vehicles = Container()  # 
        self.reward_record = dict()
        self.veh_waiting_clock = dict()
        self.veh_waiting_juncs = dict()
        self.veh_name_mapping_table = dict()
        self.conflict_vehids = []

        self.custom_collided_ids = set()

        # env level
        self._step = 0
        self.previous_obs = {}
        self.previous_action = {}
        self.previous_reward = {}
        self.previous_dones = {}
        self.pre_dir_info = {}
        self.pre_TTI_dic = {}

        # occupancy map
        self.inner_lane_obs = dict()
        self.inner_lane_occmap = dict()
        self.inner_lane_newly_enter = dict()
        for JuncID in self.junction_list:
            self.inner_lane_obs[JuncID] = dict()
            self.inner_lane_newly_enter[JuncID] = dict()
            self.inner_lane_occmap[JuncID] = dict()
            for keyword in self.keywords_order:
                self.inner_lane_obs[JuncID][keyword] = []
                self.inner_lane_newly_enter[JuncID][keyword] = []
                self.inner_lane_occmap[JuncID][keyword] = [0 for _ in range(10)]

        # vehicle queue and control queue
        self.control_queue = dict()
        self.control_queue_waiting_time = dict()
        self.queue = dict()
        self.queue_waiting_time = dict()
        self.head_of_control_queue = dict()
        self.inner_speed = dict()
        for JuncID in self.junction_list:
            self.control_queue[JuncID] = dict()
            self.control_queue_waiting_time[JuncID] = dict()
            self.queue[JuncID] = dict()
            self.queue_waiting_time[JuncID] = dict()
            self.head_of_control_queue[JuncID] = dict()
            self.inner_speed[JuncID] = []
            for keyword in self.keywords_order:
                self.control_queue[JuncID][keyword] = []
                self.control_queue_waiting_time[JuncID][keyword] = []
                self.queue[JuncID][keyword] = []
                self.queue_waiting_time[JuncID][keyword] = []
                self.head_of_control_queue[JuncID][keyword] = []

        ## global reward related
        self.previous_global_waiting = dict()
        for JuncID in self.junction_list:
            self.previous_global_waiting[JuncID] = dict()
            for keyword in self.keywords_order:
                self.previous_global_waiting[JuncID][keyword] = 0
                self.previous_global_waiting[JuncID]['sum'] = 0

        ## data monitor
        self.monitor = DataMonitor(self)

        self.deadlock_status = {}
        for JuncID in self.junction_list:
            self.deadlock_status[JuncID] = {
                'active': False,  # 当前是否处于死锁干预状态
                'victim_directions': [],  # 被选为“牺牲”的方向
                'counter': 0  # 死锁条件持续时间的计数器
            }

        self._print_debug('init_env')

    def get_avg_wait_time(self, JuncID, Keyword, mode='all'):
        ## mode = all, rv
        if mode == 'all':
            return np.mean(np.array(self.queue_waiting_time[JuncID][Keyword])) if len(
                self.queue_waiting_time[JuncID][Keyword]) > 0 else 0
        elif mode == 'rv':
            return np.mean(np.array(self.control_queue_waiting_time[JuncID][Keyword])) if len(
                self.control_queue_waiting_time[JuncID][Keyword]) > 0 else 0
        else:
            print('Error Mode in Queue Waiting time Calculation')
            return 0

    def get_queue_len(self, JuncID, Keyword, mode='all'):
        ## mode = all, rv
        if mode == 'all':
            return len(self.queue[JuncID][Keyword])
        elif mode == 'rv':
            return len(self.control_queue[JuncID][Keyword])
        else:
            print('Error Mode in Queue Length Calculation')
            return 0

    def rotated_keywords_order(self, veh):
        if veh.road_id[0] != ':':
            facing_junction_id = self.map.get_facing_intersection(veh.road_id)
            if len(facing_junction_id) == 0:
                print('error in rotating')
                return self.keywords_order
            else:
                dir, label = self.map.qurey_edge_direction(veh.road_id, veh.lane_index)
                if not label:
                    print("error in qurey lane direction and edge lable")
                    return self.keywords_order
                else:
                    ego_keyword = label + dir
                    index = self.keywords_order.index(ego_keyword)
                    rotated_keyword = []
                    for i in range(len(self.keywords_order)):
                        rotated_keyword.extend([self.keywords_order[(i + index) % (len(self.keywords_order) - 1)]])
                    return rotated_keyword
        else:
            for ind in range(len(veh.road_id)):
                if veh.road_id[len(veh.road_id) - 1 - ind] == '_':
                    break
            last_dash_ind = len(veh.road_id) - 1 - ind
            facing_junction_id = veh.road_id[1:last_dash_ind]
            dir, label = self.map.qurey_inner_edge_direction(veh.road_id, veh.lane_index)
            if not label:
                print("error in qurey lane direction and edge lable")
                return self.keywords_order
            else:
                ego_keyword = label + dir
                index = self.keywords_order.index(ego_keyword)
                rotated_keyword = []
                for i in range(len(self.keywords_order)):
                    rotated_keyword.extend([self.keywords_order[(i + index) % (len(self.keywords_order) - 1)]])
                return rotated_keyword

    def change_conflict_mechanism_type(self, new_type):
        if not new_type in ['off', 'flexible', 'standard']:
            return False
        else:
            self.conflict_resolve_mechanism_type = new_type
            return True

    def change_veh_route(self, veh_id, route):
        ## route should be a list of edge id
        self.sumo_interface.set_veh_route(veh_id, route)

    def change_rl_prob(self, rl_prob):
        changed_list = []
        if rl_prob < self.default_rl_prob:
            for veh in self.rl_vehicles:
                # route = self.routes[tc.vehicle.getRouteID(veh_id)]
                if random.random() > (rl_prob / self.default_rl_prob):
                    changed_list.extend([deepcopy(veh)])
                    self.vehicles[veh.id].type = 'IDM'
                    self.sumo_interface.set_color(self.vehicles[veh.id], WHITE)
            for veh in changed_list:
                self.rl_vehicles.pop(veh.id)
            self.change_default_spawn_rl_prob(rl_prob)
        else:
            self.change_default_spawn_rl_prob(rl_prob)
        return changed_list

    def change_default_spawn_rl_prob(self, prob):
        self.default_rl_prob = prob

    def change_spawn_rl_prob(self, edge_id, prob):
        self.spawn_rl_prob[edge_id] = prob

    def conflict_predetection(self, junc, ori):
        # detect potential conflict, refer to conflict resolving mechanism
        # input: junc:junction id, ori: moving direction
        # output: True: conflict or potential conflict, False: no conflict detected
        allowing_ori = [ori]
        for pair_set in UNCONFLICT_SET:
            if ori in pair_set:
                for k in pair_set:
                    if k != ori:
                        allowing_ori.extend([k])
        if self.conflict_resolve_mechanism_type == 'flexible':
            # self.previous_global_waiting[junc]['largest'] 记录最大的等待时间在哪条路，哪个方向的路
            if ori in self.previous_global_waiting[junc]['largest']:
                for key in self.inner_lane_occmap[junc].keys():
                    if max(self.inner_lane_occmap[junc][key][:3]) > 0 and key not in allowing_ori:
                        return True
            else:
                for key in self.inner_lane_occmap[junc].keys():
                    if max(self.inner_lane_occmap[junc][key][:8]) > 0 and key not in allowing_ori:
                        return True
        elif self.conflict_resolve_mechanism_type == 'standard':
            for key in self.inner_lane_occmap[junc].keys():
                if max(self.inner_lane_occmap[junc][key][:8]) > 0 and key not in allowing_ori:
                    return True
        elif self.conflict_resolve_mechanism_type == 'off':
            pass
        else:
            pass
        return False

    def virtual_id_assign(self, veh_id):
        if not veh_id in self.veh_name_mapping_table.keys():
            self.veh_name_mapping_table[veh_id] = (veh_id, False)
            return veh_id
        else:
            # print(self.veh_name_mapping_table[veh_id])
            if self.veh_name_mapping_table[veh_id][1]:
                virtual_new_id = veh_id + '@' + str(10 * random.random())
                self.veh_name_mapping_table[veh_id] = (virtual_new_id, False)
                return virtual_new_id
            else:
                return self.veh_name_mapping_table[veh_id][0]

    def convert_virtual_id_to_real_id(self, virtual_id):
        return virtual_id.split('@')[0]

    def terminate_veh(self, virtual_id):
        real_id = virtual_id.split('@')[0]
        self.veh_name_mapping_table[real_id] = (self.veh_name_mapping_table[real_id][0], True)

    def need_to_control(self, veh):
        # determine whether the vehicles is inside the control zone
        return True if self.map.check_veh_location_to_control(veh) and \
                       (self.map.edge_length(veh.road_id) - veh.laneposition) < self.control_distance \
            else False
    
    def get_distance_to_intersection(self, veh):
        return self.map.edge_length(veh.road_id) - veh.laneposition

    def norm_value(self, value_list, max, min):
        for idx in range(len(value_list)):
            value_list[idx] = value_list[idx] if value_list[idx] < max else max
            value_list[idx] = value_list[idx] if value_list[idx] > min else min
        return np.array(value_list) / max

    def reward_compute(self, rl_veh, waiting_lst, action, junc, ori, safe_reward=0, collision=0):

        self.reward_record[rl_veh.id] = dict()

        # efficiency reward
        action_reward = action / self.max_acc
        rank = rank_of_first(waiting_lst)
        efficiency_reward = (rank - 5.5) * action_reward

        # risk reward
        risk_reward = -3 * safe_reward

        # safe reward
        if collision == 1:
            collision_reward = -10*action_reward
        else:
            collision_reward = 10*action_reward

        # total reward
        total_reward = efficiency_reward + risk_reward + collision_reward

        self.reward_record[rl_veh.id]['efficiency'] = efficiency_reward
        self.reward_record[rl_veh.id]['safety'] = risk_reward
        self.reward_record[rl_veh.id]['total'] = total_reward

        return total_reward
        
    def _traffic_light_program_update(self):
        if self._step > self.traffic_light_program['disable_light_start']:
            self.sumo_interface.disable_all_trafficlight(self.traffic_light_program['disable_state'])

    def compute_max_len_of_control_queue(self, JuncID):
        control_queue_len = []
        junc_info = self.control_queue[JuncID]
        for keyword in self.keywords_order:
            control_queue_len.extend([len(junc_info[keyword])])
        return np.array(control_queue_len).max()

    def _compute_total_num_control_queue(self, junc_info):
        control_queue_len = []
        for keyword in self.keywords_order:
            control_queue_len.extend([len(junc_info[keyword])])
        return sum(control_queue_len)

    def _update_obs(self):
        # clear the queues
        for JuncID in self.junction_list:
            self.inner_speed[JuncID] = []
            for keyword in self.keywords_order:
                self.control_queue[JuncID][keyword] = []
                self.queue[JuncID][keyword] = []
                self.head_of_control_queue[JuncID][keyword] = []
                self.control_queue_waiting_time[JuncID][keyword] = []
                self.queue_waiting_time[JuncID][keyword] = []

        # occupancy map
        self.inner_lane_obs = dict()
        self.inner_lane_occmap = dict()
        self.inner_lane_newly_enter = dict()
        for JuncID in self.junction_list:
            self.inner_lane_obs[JuncID] = dict()
            self.inner_lane_newly_enter[JuncID] = dict()
            self.inner_lane_occmap[JuncID] = dict()
            for keyword in self.keywords_order:
                self.inner_lane_obs[JuncID][keyword] = []
                self.inner_lane_newly_enter[JuncID][keyword] = []
                self.inner_lane_occmap[JuncID][keyword] = [0 for _ in range(10)]

        for veh in self.vehicles:
            if len(veh.road_id) == 0:
                ## avoid invalid vehicle information
                continue
            if veh.road_id[0] == ':':
                ## inside intersection: update inner obs and occmap
                direction, edge_label = self.map.qurey_inner_edge_direction(veh.road_id, veh.lane_index)
                for ind in range(len(veh.road_id)):
                    if veh.road_id[len(veh.road_id) - 1 - ind] == '_':
                        break
                last_dash_ind = len(veh.road_id) - 1 - ind
                if edge_label and veh.road_id[1:last_dash_ind] in self.junction_list:
                    self.inner_lane_obs[veh.road_id[1:last_dash_ind]][edge_label + direction].extend([veh])
                    self.inner_lane_occmap[veh.road_id[1:last_dash_ind]][edge_label + direction][
                        min(round(10 * veh.laneposition / self.map.edge_length(veh.road_id)), 9)] = 1
                    if veh not in self.prev_inner[veh.road_id[1:last_dash_ind]][edge_label + direction]:
                        self.inner_lane_newly_enter[veh.road_id[1:last_dash_ind]][edge_label + direction].extend([veh])
                    self.inner_speed[veh.road_id[1:last_dash_ind]].extend([veh.speed])
            else:
                ## update waiting time
                JuncID, keyword = self.map.get_veh_moving_direction(veh)
                accumulating_waiting = veh.wait_time
                if len(JuncID) > 0:
                    if veh.id not in self.veh_waiting_juncs.keys():
                        self.veh_waiting_juncs[veh.id] = dict()
                        self.veh_waiting_juncs[veh.id][JuncID] = accumulating_waiting
                    else:
                        prev_wtm = 0
                        for prev_JuncID in self.veh_waiting_juncs[veh.id].keys():
                            if prev_JuncID != JuncID:
                                prev_wtm += self.veh_waiting_juncs[veh.id][prev_JuncID]
                        if accumulating_waiting - prev_wtm >= 0:
                            self.veh_waiting_juncs[veh.id][JuncID] = accumulating_waiting - prev_wtm
                        else:
                            self.veh_waiting_juncs[veh.id][JuncID] = accumulating_waiting

                ## updating control queue and waiting time of queue
                if self.map.get_distance_to_intersection(veh) <= self.control_zone_length:
                    self.queue[JuncID][keyword].extend([veh])
                    self.queue_waiting_time[JuncID][keyword].extend([self.veh_waiting_juncs[veh.id][JuncID]])
                    if veh.type == 'RL':
                        self.control_queue[JuncID][keyword].extend([veh])
                        self.control_queue_waiting_time[JuncID][keyword].extend(
                            [self.veh_waiting_juncs[veh.id][JuncID]])

        ## update previous global waiting for next step reward calculation
        for JuncID in self.junction_list:
            weighted_sum = 0
            largest = 0
            for Keyword in self.keywords_order:
                control_queue_length = self.get_queue_len(JuncID, Keyword, 'rv')
                waiting_time = self.get_avg_wait_time(JuncID, Keyword, 'rv')
                self.previous_global_waiting[JuncID][Keyword] = waiting_time
                if waiting_time >= largest:
                    self.previous_global_waiting[JuncID]['largest'] = [Keyword]
                    largest = waiting_time
                weighted_sum += waiting_time
            self.global_obs[JuncID] = (self.previous_global_waiting[JuncID]['sum'] - weighted_sum) / (
                        self.previous_global_waiting[JuncID]['sum'] * 10 + EPSILON)
            if self.global_obs[JuncID] < -1:
                self.global_obs[JuncID] = -1
            if self.global_obs[JuncID] > 1:
                self.global_obs[JuncID] = 1
            self.previous_global_waiting[JuncID]['sum'] = weighted_sum

        for JuncID in self.junction_list:
            self._check_and_manage_deadlock(JuncID)

    def _check_and_manage_deadlock(self, JuncID):
        """为单个交叉口检测并管理死锁状态"""
        if not self.use_deadlock_resolver:
            return

        status = self.deadlock_status[JuncID]

        # 检查是否可以解除死锁状态
        if status['active']:
            all_queues_cleared = True
            for direction in status['victim_directions']:
                # 如果牺牲方向的队列已经变短，可以考虑解除
                if self.get_queue_len(JuncID, direction, 'rv') > 2:
                    all_queues_cleared = False
                    break
            if all_queues_cleared:
                status['active'] = False
                status['victim_directions'] = []
                # print(f"[{self._step}] Deadlock resolved at {JuncID}")
            return  # 在干预期间，不进行新的检测

        # --- 死锁检测逻辑 ---
        # 规则1: 检测高压状态 (多个冲突方向队列都很长)
        # 这里简化为检测所有方向，实际应用中可以构建冲突矩阵
        long_queue_directions = []
        for keyword in self.keywords_order:
            if self.get_queue_len(JuncID, keyword, 'rv') >= self.deadlock_thresholds['queue_len']:
                long_queue_directions.append(keyword)

        if len(long_queue_directions) < 2:  # 至少要有两个长队列才可能死锁
            status['counter'] = 0  # 重置计数器
            return

        # 规则3: 检测循环等待 (队首车长时间不动)
        is_circular_wait = True
        for direction in long_queue_directions:
            # 确保队列不为空
            if not self.control_queue[JuncID][direction]:
                is_circular_wait = False
                break

            head_vehicle = self.control_queue[JuncID][direction][0]
            if head_vehicle.speed > self.deadlock_thresholds['speed'] or \
                    head_vehicle.wait_time < self.deadlock_thresholds['wait_time']:
                is_circular_wait = False
                break

        if is_circular_wait:
            status['counter'] += 1
        else:
            status['counter'] = 0  # 条件中断，重置计数器
            return

        # --- 死锁决策逻辑 ---
        if status['counter'] >= self.deadlock_thresholds['persistence']:
            # print(f"[{self._step}] Deadlock DETECTED at {JuncID} in directions {long_queue_directions}")
            status['active'] = True
            status['counter'] = 0

            # 决策: 选择一个“牺牲”方向 (总等待时间最短的)
            min_wait_time = float('inf')
            victim = None
            for direction in long_queue_directions:
                # 使用所有车辆的平均等待时间做决策更稳定
                avg_wait = self.get_avg_wait_time(JuncID, direction, 'all')
                if avg_wait < min_wait_time:
                    min_wait_time = avg_wait
                    victim = direction

            if victim:
                status['victim_directions'] = [victim]
                # print(f"[{self._step}] Victim selected for {JuncID}: {victim}")

    def step_once(self, action={}, safe_reward_dict=None):
        self._print_debug('step')
        self.new_departed = set()  # 可能是新离开的车辆
        self.sumo_interface.set_max_speed_all(10)  # 设置默认车辆最大速度
        self._traffic_light_program_update()  # 信号灯设置为绿色
        # check if the action input is valid
        if not (isinstance(action, dict) and len(action) == len(self.previous_obs) - sum(dict_tolist(self.previous_dones))):
            print('error!! action dict is invalid')
            return dict()

        # execute action in the sumo env 执行动作在sumo中，改变sumo里面东西的状态
        current_sumo_veh_ids = self.sumo_interface.tc.vehicle.getIDList()
        for virtual_id in action.keys():

            veh_id = self.convert_virtual_id_to_real_id(virtual_id)  # 根据实际的id转化为对应的车辆id

            if veh_id not in current_sumo_veh_ids:
                # print("test1")
                continue
            action_value = action[virtual_id]
            
            # 处理连续动作：直接将动作值作为加速度使用
            # 确保加速度在允许范围内
            acceleration = float(action_value)  # 确保是浮点数
            acceleration = max(self.min_acc, min(self.max_acc, acceleration))  # 限制在[-3, 3]范围内
            
            # 根据动作值设置对应的加速度
            self.sumo_interface.accl_control(self.rl_vehicles[veh_id], acceleration)
            
            # 根据加速度设置车辆颜色以便观察
            if acceleration > 0:
                self.sumo_interface.set_color(self.rl_vehicles[veh_id], GREEN)  # 加速-绿色
            elif acceleration < 0:
                self.sumo_interface.set_color(self.rl_vehicles[veh_id], RED)    # 减速-红色


        # sumo step sumo运行
        self.sumo_interface.step()

        # gathering states from sumo 从sumo中收集状态
        sim_res = self.sumo_interface.get_sim_info()

        # setup for new departed vehicles 新到达车辆的设置（要离开的车辆）
        for veh_id in sim_res.departed_vehicles_ids:
            self.sumo_interface.subscribes.veh.subscribe(veh_id)  # 订阅属性值，一旦订阅，系统就会在一定时间更新属性值
            length = self.sumo_interface.tc.vehicle.getLength(veh_id)
            route = self.sumo_interface.tc.vehicle.getRoute(veh_id)
            road_id = self.sumo_interface.get_vehicle_edge(veh_id)
            if (road_id in self.spawn_rl_prob.keys() and random.random() < self.spawn_rl_prob[road_id]) or \
                    (random.random() < self.default_rl_prob):  # 手动设置RL车辆和IDM车辆
                self.rl_vehicles[veh_id] = veh = Vehicle(id=veh_id, type="RL", route=route, length=length)
                self.vehicles[veh_id] = veh = Vehicle(id=veh_id, type="RL", route=route, length=length, wait_time=0)
            else:
                self.vehicles[veh_id] = veh = Vehicle(id=veh_id, type="IDM", route=route, length=length, wait_time=0)

            # self.sumo_interface.set_color(veh, WHITE if veh.type == "IDM" else RED)  # 如果是IDM车辆，那么就是白色，否则是红色，车身颜色

            self.new_departed.add(veh)

        self.new_arrived = {self.vehicles[veh_id] for veh_id in sim_res.arrived_vehicles_ids}
        self.new_collided = {self.vehicles[veh_id] for veh_id in sim_res.colliding_vehicles_ids}
        self.new_arrived -= self.new_collided  # Don't count collided vehicles as "arrived"

        for veh in self.new_arrived:
            if veh.type == 'RL':
                self.rl_vehicles.pop(veh.id)
            self.vehicles.pop(veh.id)

        junction_colliding_vehicles = set()
        for veh_id in sim_res.colliding_vehicles_ids:
            current_road_id = self.sumo_interface.get_vehicle_edge(veh_id)
            if current_road_id.startswith(':'):
                JuncID, ego_dir = self.map.get_veh_moving_direction(self.vehicles[veh_id])
                if JuncID in self.junction_list and veh_id in self.vehicles:
                    junction_colliding_vehicles.add(self.vehicles[veh_id])

        # for veh_id, veh in self.vehicles.items():
        #     if veh not in self.new_collided and veh not in self.new_arrived: # 存在性判断
        #         current_road_id = self.sumo_interface.get_vehicle_edge(veh_id)
        #         if current_road_id.startswith(':'):
        #             JuncID, ego_dir = self.map.get_veh_moving_direction(self.vehicles[veh_id])
        #             if JuncID in self.junction_list and veh.speed < 0.1:
        #                 junction_colliding_vehicles.add(veh)
        #                 # self.sumo_interface.tc.vehicle.remove(veh_id)

        self.new_collided = junction_colliding_vehicles

        self.conflict_vehids = self.new_collided
        # 更新模型数据
        if self.use_safety_predictor:
            for veh in self.new_arrived:
                self.risk_predictor.buffer.update_data(veh.id, 0)
            for veh in self.new_collided:
                self.risk_predictor.buffer.update_data(veh.id, 1)

        # for veh in self.new_collided:
        #     if veh.type == 'RL':
        #         self.rl_vehicles.pop(veh.id)
        #     self.vehicles.pop(veh.id)
        # remove arrived vehicles from Env 把离开的车辆从env中移除

        self._print_debug('before updating vehicles')
        # update vehicles' info for Env 更新车辆信息
        current_sumo_veh_ids = self.sumo_interface.tc.vehicle.getIDList()
        for veh_id, veh in self.vehicles.items():
            if veh_id not in current_sumo_veh_ids:
                continue
            veh.prev_speed = veh.get('speed', None)  # 把以前的速度prev_speed，更新为当前的速度，当前的变成新的
            veh.update(self.sumo_interface.subscribes.veh.get(veh_id))  # 更新veh的其他参数
            if veh.type == 'RL':
                self.rl_vehicles[veh_id].update(self.sumo_interface.subscribes.veh.get(veh_id))
            wt, _ = self.sumo_interface.get_veh_waiting_time(veh)
            if wt > 0:
                self.vehicles[veh_id].wait_time += 1

        ## update obs
        self._update_obs()

        TTI_dict = {}
        for rl_veh in self.rl_vehicles:
            if self.need_to_control(rl_veh):
                virtual_id = self.virtual_id_assign(rl_veh.id)
                JuncID, ego_dir = self.map.get_veh_moving_direction(rl_veh)
                collision = 0
                if self.conflict_predetection(JuncID, ego_dir):
                    collision = 1
                TTI_dict[virtual_id] = collision

        new_obs = {}
        # 获取当前的状态空间
        for virtual_id, val in self.pre_dir_info.items():
            JuncID, rotated_keywords = val[0], val[1]
            obs_control_queue_length = []
            obs_waiting_lst = []
            obs_inner_lst = []
            control_queue_max_len = self.compute_max_len_of_control_queue(JuncID) + EPSILON
            for keyword in rotated_keywords:
                obs_control_queue_length.extend([self.get_queue_len(JuncID, keyword, 'rv') / control_queue_max_len])
                obs_waiting_lst.extend([self.get_avg_wait_time(JuncID, keyword, 'rv')])
                obs_inner_lst.append(self.inner_lane_occmap[JuncID][keyword])
            obs_waiting_lst = self.norm_value(obs_waiting_lst, self.max_wait_time, 0)

            veh_speed = self.rl_vehicles[val[2]].speed
            speed_state = veh_speed / self.max_speed

            distance_to_intersection = self.get_distance_to_intersection(self.rl_vehicles[val[2]]) / self.control_distance
            if distance_to_intersection < 0:
                distance_to_intersection = 0
            new_obs[virtual_id] = self.check_obs_constraint(np.concatenate(
                ([distance_to_intersection], [speed_state], obs_control_queue_length, np.array(obs_waiting_lst), np.reshape(np.array(obs_inner_lst), (80,)))))

        obs = {}
        rewards = {}
        dones = {}
        dir_info = {}
        current_sumo_veh_ids = self.sumo_interface.tc.vehicle.getIDList()
        for rl_veh in self.rl_vehicles:
            virtual_id = self.virtual_id_assign(rl_veh.id)  # id转换
            if rl_veh.id not in current_sumo_veh_ids:
                continue
            # 获取当前车辆的速度
            current_speed = self.rl_vehicles[rl_veh.id].speed
            # 计算速度one-hot编码
            speed_state = current_speed/self.max_speed

            # print("id:",virtual_id,rl_veh.id)
            if len(rl_veh.road_id) == 0:  # 车没了
                if virtual_id in action.keys():
                    ## collision occured and teleporting, I believe it should be inside the intersection
                    obs[virtual_id] = self.check_obs_constraint(self.previous_obs[virtual_id])
                    # print(obs[virtual_id])
                    dones[virtual_id] = True
                    rewards[virtual_id] = -1
                    self.terminate_veh(virtual_id)
                    continue
                else:
                    ## then do nothing
                    continue

            JuncID, ego_dir = self.map.get_veh_moving_direction(rl_veh)
            if len(JuncID) == 0 or JuncID not in self.junction_list:
                # skip the invalid JuncID
                continue

            obs_control_queue_length = []
            obs_waiting_lst = []
            obs_inner_lst = []
            control_queue_max_len = self.compute_max_len_of_control_queue(JuncID) + EPSILON
            if self.need_to_control(rl_veh):

                dir_info[virtual_id] = (JuncID, self.rotated_keywords_order(rl_veh), rl_veh.id)
                for keyword in self.rotated_keywords_order(rl_veh):
                    obs_control_queue_length.extend([self.get_queue_len(JuncID, keyword, 'rv') / control_queue_max_len])
                    obs_waiting_lst.extend([self.get_avg_wait_time(JuncID, keyword, 'rv')])
                    obs_inner_lst.append(self.inner_lane_occmap[JuncID][keyword])

                obs_waiting_lst = self.norm_value(obs_waiting_lst, self.max_wait_time, 0)
                if virtual_id in self.previous_obs:
                    pre_obs_waiting_lst = self.previous_obs[virtual_id][12:20]
                else:
                    pre_obs_waiting_lst = np.zeros(8)
                if virtual_id in self.pre_TTI_dic:
                    collision = self.pre_TTI_dic[virtual_id]
                else:
                    collision = 0
                if virtual_id in action.keys():
                    ## reward
                    if not safe_reward_dict:
                        rewards[virtual_id] = self.reward_compute(rl_veh, pre_obs_waiting_lst, action[virtual_id], JuncID,
                                                                ego_dir, 0, collision)
                    else:
                        rewards[virtual_id] = self.reward_compute(rl_veh, pre_obs_waiting_lst, action[virtual_id], JuncID,
                                                                ego_dir, safe_reward_dict[virtual_id], collision)
                # 获取车辆到交叉口的距离
                distance_to_intersection = self.get_distance_to_intersection(rl_veh) / self.control_distance
                obs[virtual_id] = self.check_obs_constraint(np.concatenate(
                    ([distance_to_intersection], [speed_state], obs_control_queue_length, np.array(obs_waiting_lst), np.reshape(np.array(obs_inner_lst), (80,)))))
                dones[virtual_id] = False
            elif virtual_id in action.keys():
                ## update reward for the vehicle already enter intersection
                if rl_veh.road_id[0] == ':':
                    ## inside the intersection
                    for keyword in self.rotated_keywords_order(rl_veh):
                        obs_control_queue_length.extend(
                            [self.get_queue_len(JuncID, keyword, 'rv') / control_queue_max_len])
                        obs_waiting_lst.extend([self.get_avg_wait_time(JuncID, keyword, 'rv')])
                        obs_inner_lst.append(self.inner_lane_occmap[JuncID][keyword])
                    obs_waiting_lst = self.norm_value(obs_waiting_lst, self.max_wait_time, 0)
                    if virtual_id in self.previous_obs:
                        pre_obs_waiting_lst = self.previous_obs[virtual_id][12:20]
                    else:
                        pre_obs_waiting_lst = np.zeros(8)
                    if virtual_id in self.pre_TTI_dic:
                        collision = self.pre_TTI_dic[virtual_id]
                    else:
                        collision = 0
                    if not safe_reward_dict:
                        rewards[virtual_id] = self.reward_compute(rl_veh, pre_obs_waiting_lst, action[virtual_id], JuncID,
                                                              ego_dir, 0, collision)
                    else:
                        rewards[virtual_id] = self.reward_compute(rl_veh, pre_obs_waiting_lst, action[virtual_id], JuncID,
                                                              ego_dir, safe_reward_dict[virtual_id],collision)
                    dones[virtual_id] = True
                    obs[virtual_id] = self.check_obs_constraint(np.concatenate(
                        ([0], [speed_state], obs_control_queue_length, np.array(obs_waiting_lst), np.reshape(np.array(obs_inner_lst),(80,)))))
                    self.terminate_veh(virtual_id)
                else:
                    ## change to right turn lane and no need to control
                    for keyword in self.rotated_keywords_order(rl_veh):
                        obs_control_queue_length.extend(
                            [self.get_queue_len(JuncID, keyword, 'rv') / control_queue_max_len])
                        obs_waiting_lst.extend([self.get_avg_wait_time(JuncID, keyword, 'rv')])
                        obs_inner_lst.append(self.inner_lane_occmap[JuncID][keyword])
                    obs_waiting_lst = self.norm_value(obs_waiting_lst, self.max_wait_time, 0)
                    rewards[virtual_id] = 0
                    dones[virtual_id] = True
                    obs[virtual_id] = self.check_obs_constraint(np.concatenate(([0], [speed_state], obs_control_queue_length,
                                                                                np.array(obs_waiting_lst),
                                                                                np.reshape(np.array(obs_inner_lst),
                                                                                           (80,)))))
                    self.terminate_veh(virtual_id)
        dones['__all__'] = False
        infos = {}
        truncated = {}
        truncated['__all__'] = False
        if self._step >= self._max_episode_steps:
            for key in dones.keys():
                truncated[key] = True
        self._step += 1
        self.previous_obs, self.previous_reward, self.previous_action, self.previous_dones, self.prev_inner \
            = deepcopy(obs), deepcopy(rewards), deepcopy(action), deepcopy(dones), deepcopy(self.inner_lane_obs)
        self.pre_dir_info = deepcopy(dir_info)
        self.pre_TTI_dic = deepcopy(TTI_dict)
        
        self.monitor.step(self)

        self.conflict_vehids = []
        self._print_debug('finish process step')
        if len(dict_tolist(rewards)) > 0 and self.print_debug:
            print('avg reward: ' + str(np.array(dict_tolist(rewards)).mean()) + ' max reward: ' + str(
                np.array(dict_tolist(rewards)).max()) + ' min reward: ' + str(np.array(dict_tolist(rewards)).min()))
        # obs有队列长度8，等待队列8，inner_list三部分组成
        return obs, rewards, dones, truncated, infos, new_obs

    def step(self, action={}, safe_reward_dict=None):
        # if len(action) == 0:
        #     print("empty action")

        obs, rewards, dones, truncated, infos, old_obs = self.step_once(action, safe_reward_dict)

        return obs, rewards, dones, truncated, infos, old_obs

    def reset(self, *, seed=None, options=None):  # 环境初始化
        self._print_debug('reset')
        # soft reset
        while not self.sumo_interface.reset_sumo():  # sumo重置
            pass

        if self.rl_prob_list:  # 设置的RV车辆的比例
            self.default_rl_prob = random.choice(self.rl_prob_list)
            print("new RV percentage = " + str(self.default_rl_prob))

        self.init_env()  # 环境初始化
        obs = {}  # 状态观测
        if options:
            if options['mode'] == 'HARD':
                obs, _, _, _, infos, new_obs = self.step_once()
                return obs, infos

        while len(obs) == 0:
            obs, _, _, _, infos, new_obs = self.step_once()  # 走一步
        return obs, infos

    def close(self):
        ## close env
        self.sumo_interface.close()


    def check_obs_constraint(self, obs):
        if not self.observation_space.contains(obs):
            obs = np.asarray([x if x >= self.observation_space.low[0] else self.observation_space.low[0] for x in obs] \
                             , dtype=self.observation_space.dtype)
            obs = np.asarray([x if x <= self.observation_space.high[0] else self.observation_space.high[0] for x in obs] \
                             , dtype=self.observation_space.dtype)
            if not self.observation_space.contains(obs):
                print('dddd')
                raise ValueError(
                    "Observation is invalid, got {}".format(obs)
                )
        return obs
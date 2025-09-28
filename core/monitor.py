import numpy as np
import pickle
import math


class DataMonitor(object):
    def __init__(self, env) -> None:
        self.junction_list = env.junction_list
        self.keywords_order = env.keywords_order
        self.clear_data()

    def clear_data(self):
        self.conduct_traj_recorder()
        self.conduct_data_recorder()

    def conduct_traj_recorder(self):
        self.traj_record = dict()
        for JuncID in self.junction_list:
            self.traj_record[JuncID] = dict()
            for Keyword in self.keywords_order:
                self.traj_record[JuncID][Keyword] = dict()
        self.max_t = 0
        self.max_x = 0

    def conduct_data_recorder(self):
        self.data_record = dict()
        self.conflict_rate = []
        for JuncID in self.junction_list:
            self.data_record[JuncID] = dict()
            for Keyword in self.keywords_order :
                self.data_record[JuncID][Keyword] = dict()
                self.data_record[JuncID][Keyword]['t'] = [i for i in range(5000)]
                self.data_record[JuncID][Keyword]['queue_wait'] = np.zeros(5000)
                self.data_record[JuncID][Keyword]['queue_length'] = np.zeros(5000)
                self.data_record[JuncID][Keyword]['control_queue_wait'] = np.zeros(5000)
                self.data_record[JuncID][Keyword]['control_queue_length'] = np.zeros(5000)
                self.data_record[JuncID][Keyword]['throughput_av'] = np.zeros(5000)
                self.data_record[JuncID][Keyword]['throughput'] = np.zeros(5000)
                self.data_record[JuncID][Keyword]['throughput_hv'] = np.zeros(5000)
                self.data_record[JuncID][Keyword]['conflict'] = np.zeros(5000)
                self.data_record[JuncID][Keyword]['global_reward'] = np.zeros(5000)

                self.data_record[JuncID][Keyword]['avg_speed'] = np.zeros(5000)
        self.total_data = dict()
        self.total_data['max_queue_length'] = 0
        self.total_data['avg_queue_length'] = 0
        self.total_data['queue_wait'] = 0
        self.total_data['conflict'] = 0
        self.total_data['veh_num'] = 0
                

    def step(self, env):
        t = env.env_step

        all_speeds = [veh.speed for veh in env.vehicles]
        # 如果存在车辆，则计算平均速度，否则为0
        current_avg_speed = np.mean(all_speeds) if len(all_speeds) > 0 else 0

        for JuncID in self.junction_list:
            for Keyword in self.keywords_order:
                self.data_record[JuncID][Keyword]['queue_length'][t] = env.get_queue_len(JuncID, Keyword, 'all')
                self.data_record[JuncID][Keyword]['queue_wait'][t] = env.get_avg_wait_time(JuncID, Keyword, 'all')
                self.data_record[JuncID][Keyword]['control_queue_length'][t] = env.get_queue_len(JuncID, Keyword, 'rv')
                self.data_record[JuncID][Keyword]['control_queue_wait'][t] = env.get_avg_wait_time(JuncID, Keyword, 'rv')
                self.data_record[JuncID][Keyword]['throughput'][t] = len(env.inner_lane_newly_enter[JuncID][Keyword])
                self.data_record[JuncID][Keyword]['conflict'][t] = len(env.conflict_vehids)
                self.data_record[JuncID][Keyword]['global_reward'][t] = env.global_obs[JuncID]

                self.data_record[JuncID][Keyword]['avg_speed'][t] = current_avg_speed

                self.total_data['max_queue_length'] = max(self.total_data['max_queue_length'], env.get_queue_len(JuncID, Keyword, 'all'))
                self.total_data['avg_queue_length'] += env.get_queue_len(JuncID, Keyword, 'all')
                self.total_data['queue_wait'] += env.get_avg_wait_time(JuncID, Keyword, 'all')
                self.total_data['veh_num'] += len(env.inner_lane_newly_enter[JuncID][Keyword])
        self.total_data['conflict'] += len(env.conflict_vehids)

        

        self.conflict_rate.extend(
            [len(env.conflict_vehids)/len(env.previous_action) if len(env.previous_action) else 0]
            )

    def evaluate(self, min_step = 500, max_step = 1000):
        total_wait = []
        for JuncID in self.junction_list:
            for keyword in self.keywords_order:
                avg_wait = np.mean(self.data_record[JuncID][keyword]['queue_wait'][min_step:max_step])
                total_wait.extend([avg_wait])
                print("Avg waiting time at" + JuncID +" "+keyword+": "+str(avg_wait))
            print("Total avg wait time at junction "+JuncID+": " +str(np.mean(total_wait)))
        

    def eval_traffic_flow(self, JuncID, time_range):
        inflow_intersection = []
        for t in range(time_range[0], time_range[1]):
            inflow_intersection.extend([0])
            for Keyword in self.keywords_order:
                 inflow_intersection[-1] += self.data_record[JuncID][Keyword]['throughput'][t]
        return inflow_intersection, max(inflow_intersection), sum(inflow_intersection)/len(inflow_intersection)

    def save_to_pickle(self, file_name):
        saved_dict = {'data_record':self.data_record, 'junctions':self.junction_list, 'keyword':self.keywords_order}
        with open(file_name, "wb") as f:
            pickle.dump(saved_dict, f)
    
    def get_data(self, env):
        t = env.env_step
        data_dict = {"max_quene_length":0, "avg_quene_length":0, "quene_wait":0, "conflict":0, "conflict_rate":0}
        data_dict['max_quene_length'] = self.total_data['max_queue_length']
        data_dict['avg_quene_length'] = self.total_data['avg_queue_length'] / (len(self.junction_list) * len(self.keywords_order) * t)
        data_dict['quene_wait'] = self.total_data['queue_wait'] / (len(self.junction_list) * len(self.keywords_order) * t)
        data_dict['conflict'] = self.total_data['conflict']
        data_dict['conflict_rate'] = self.total_data['conflict'] / self.total_data['veh_num']

        first_junc = self.junction_list[0]
        first_keyword = self.keywords_order[0]
        # Get all speed data up to the current step `t`
        speed_data_so_far = self.data_record[first_junc][first_keyword]['avg_speed'][:t]
        # Calculate the mean of the recorded per-step average speeds
        data_dict['avg_speed'] = np.mean(speed_data_so_far)

        return data_dict

import time

from btgym import BehaviorTree
import btgym
from btgym.utils import ROOT_PATH
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from btgym.envs.virtualhometext.exec_lib._base.VHTAction_small import VHTAction_small
from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction
from btgym.algos.bt_autogen.tools import state_transition
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process
import random
import numpy as np
import pandas as pd

seed = 0
random.seed(seed)
np.random.seed(seed)

from btgym.utils.tools import collect_action_nodes
from btgym.utils.read_dataset import read_dataset

# env_path = f"{ROOT_PATH}/../test/dataset/environment.txt"
# env_dic = read_environment(env_path,style=True)

#########################


import pickle

# def refresh_VHT_samll_data():
# 读入数据集合
# data_path = f"{ROOT_PATH}/../test/dataset/data0429.txt"
# data = read_dataset(data_path)
# data_path = f"{ROOT_PATH}/../test/dataset/dataset_noenv.txt"
data_path = f"{ROOT_PATH}/../test/dataset/data_cys.txt"
data = read_dataset(data_path)

data = data[:10]

data_num = len(data)
print(f"导入 {data_num} 条数据")
print(data[0])

# 数据集中涉及的所有物体集合
objs = set()
for d in data:
    objs |= set(d['Vital Objects'])

categories = ['SURFACES', 'SITTABLE', 'CAN_OPEN', 'CONTAINERS', 'GRABBABLE', 'cleaning_tools', \
              'cutting_tools', 'HAS_SWITCH', 'HAS_PLUG', 'CUTABLE', 'EATABLE', 'WASHABLE', 'RECIPIENT', \
              'POURABLE', 'DRINKABLE']
categories_objs_dic = {}
for ctg in categories:
    categories_objs_dic[ctg] = getattr(VHTAction, ctg)
    categories_objs_dic[ctg] &= objs

ctg_objs_path = f"{ROOT_PATH}/../test/BT_EXP/ctg_objs.pickle"
# 打开一个文件用于写入，注意'b'表示二进制模式
with open(ctg_objs_path, 'wb') as file:
    # 使用pickle.dump()函数将数据写入文件
    pickle.dump(categories_objs_dic, file)
################


all_start_time = time.time()

env = btgym.make("VHT-Small")
action_list = collect_action_nodes(env.behavior_lib)
all_actions_set = set(action_list)
all_actions_str_set = {act.name for act in all_actions_set}
# Easy
# goal_set = [{'IsClean(nightstand)'}]
# true_priority_act_set = {'Walk(rag)','RightGrab(rag)','Walk(nightstand)','Wipe(nightstand)'}

# Medium
# 正确的动作在数据集外面
# goal_set = [{'IsClean(nightstand)','IsOn(clock,nightstand)'}]
# true_priority_act_set = {'Walk(rag)','RightGrab(rag)','Walk(clock)','LeftGrab(clock)',\
#                          'Walk(nightstand)','Wipe(nightstand)','LeftPut(clock,nightstand)'}
# goal_set = [{'IsClean(bed)','IsOn(toy,bed)'}]
# true_priority_act_set = {'Walk(rag)','RightGrab(rag)','Walk(toy)','LeftGrab(toy)',\
#                          'Walk(bed)','Wipe(bed)','LeftPut(toy,bed)'}

# Hard
# goal_set = [{'IsIn(milk,microwave)','IsSwitchedOn(microwave)'}]
# true_priority_act_set = {"Walk(milk)", "RightGrab(milk)", "Walk(microwave)", "Open(microwave)","PlugIn(microwave)", \
#                     "RightPutIn(milk,microwave)",'SwitchOn(microwave)'}
# goal_set = [{'IsIn(milk,microwave)','IsSwitchedOn(microwave)'}]
# true_priority_act_set = {"Walk(milk)", "RightGrab(milk)", "Walk(microwave)", "Open(microwave)","PlugIn(microwave)", \
#                     "RightPutIn(milk,microwave)",'SwitchOn(microwave)'}


len_data = len(data)
# len_data = 3
# Rate range list
error_rate_range_ls = [0, 1, 5, 10, 20]
# correct_rate_range_ls = [0, 0.25, 0.5, 0.75, 1]
correct_rate_range_ls = [0,0.1,]
#   0     1          0  6       7   0.0241
#  0.2  0.8          1  5      813  2.08544
#  0.5  0.5          3  3     1579  4.50097
#  0.8  0.2          4  2     1855  5.394
#  0.9  0.1          5  1     2318  8.324
#   1    0           6  0     1831  6.0884

# Data storage dictionary
data_storage_dic = {(e, c): {'error_num': [], 'correct_num': [], \
                             'expanded_num': [], 'planning_time_total': [], \
                             'current_cost': [], 'act_steps': [], 'bt_status': []}
                    for e in error_rate_range_ls for c in correct_rate_range_ls}

for data_ind in range(len_data):
    # for data_ind in range(4,5):
    combined_string = ' & '.join(data[data_ind]['Goals'])
    goal_set = goal_transfer_str(combined_string)
    true_priority_act_set = set(act_str_process(data[data_ind]['Actions'], already_split=True))
    print("\ndata:", data_ind)
    print("goal:", goal_set)
    print("act:", true_priority_act_set)

    error_priority_act_set = all_actions_str_set - true_priority_act_set

    for error_rate in error_rate_range_ls:
        for correct_rate in correct_rate_range_ls:
    # for error_rate in [0.5]:
    #     for correct_rate in [0.25]:
            # error_rate = 0.5
            # correct_rate = 0.75

            error_num = int(len(true_priority_act_set) * error_rate)
            correct_num = int(len(true_priority_act_set) * correct_rate)
            # correct_num=len(true_priority_act_set)-error_num
            print("Error:", error_num)
            print("Correct:", correct_num)

            # 排序是为了取消随机性
            # 推荐优先级
            priority_act_ls = set()
            priority_act_ls |= set(random.sample(sorted(list(error_priority_act_set)), error_num))
            priority_act_ls |= set(random.sample(sorted(list(true_priority_act_set)), correct_num))
            priority_act_ls = sorted(list(priority_act_ls))

            print("priority_act_ls:", priority_act_ls)
            # {'LeftPut(cupcake,floor)', 'Wipe(juice)', 'PlugIn(toaster)', 'LeftPut(towel,radio)', 'Walk(bathroomcabinet)', 'LeftPut(bellpepper,kitchencounter)'}
            # priority_act_ls = {'RightGrab(pillow)', 'LeftPut(cuttingboard,kitchentable)', 'LeftPut(cuttingboard,sofa)', 'RightPut(boardgame,sofa)', 'LeftPut(cutleryknife,desk)', 'Walk(book)'}
            # priority_act_ls = []

            cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)",
                                                          "IsStanding(self)"}
            cur_cond_set |= {f'IsClose({arg})' for arg in VHTAction_small.CAN_OPEN}
            cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHTAction_small.HAS_SWITCH}
            cur_cond_set |= {f'IsUnplugged({arg})' for arg in VHTAction_small.HAS_PLUG}

            # cur_cond_set = set(env_dic[data[0]['Environment']])
            # print("cur_cond_set:",cur_cond_set)

            priority_obj_ls = []
            # 这里写错了 baseline 应该是不推荐的
            # algo = BTExpInterface(None, cur_cond_set, priority_act_ls, priority_obj_ls,\
            #                       selected_algorithm="baseline",action_list=action_list)

            # 这里应该是推荐错了怎么处理
            algo = BTExpInterface(None, cur_cond_set, priority_act_ls, priority_obj_ls, \
                                  selected_algorithm="opt", action_list=action_list)

            start_time = time.time()
            algo.process(goal_set)
            end_time = time.time()

            ptml_string, cost, expanded_num = algo.post_process()  # 后处理
            print("Expanded Conditions: ", expanded_num)
            planning_time_total = (end_time - start_time)
            print("planning_time_total:", planning_time_total)
            # print("cost_total:", cost)
            file_name = "test"
            file_path = f'./{file_name}.btml'
            with open(file_path, 'w') as file:
                file.write(ptml_string)

            # 读取执行
            bt = BehaviorTree(file_name + ".btml", env.behavior_lib)
            # bt.print()
            # bt.draw()

            env.agents[0].bind_bt(bt)
            env.reset()
            env.print_ticks = True

            # simulation and test
            # print("\n================ ")

            goal = goal_set[0]
            state = cur_cond_set
            steps = 0
            current_cost = 0
            current_tick_time = 0

            act_num = 1

            bt_status = True
            val, obj, cost, tick_time = algo.algo.bt.cost_tick(state, 0, 0)  # tick行为树，obj为所运行的行动
            # print(f"Action: {act_num}  {obj.__str__().ljust(35)}cost: {cost}")
            current_tick_time += tick_time
            current_cost += cost
            while val != 'success' and val != 'failure':
                state = state_transition(state, obj)
                val, obj, cost, tick_time = algo.algo.bt.cost_tick(state, 0, 0)
                act_num += 1
                # print(f"Action: {act_num}  {obj.__str__().ljust(35)}cost: {cost}")
                current_cost += cost
                current_tick_time += tick_time
                if (val == 'failure'):
                    # print("bt fails at step", steps)
                    bt_status = False
                    error = True
                    break
                steps += 1
                if (steps >= 500):  # 至多运行500步
                    break
            if goal <= state:
                pass
                # print("Finished!")
            else:
                bt_status = False
            print("bt_status:", bt_status)
            # print(f"一定运行了 {act_num - 1} 个动作步")
            # print("current_cost:", current_cost)
            # print("================ ")

            # 数据记录
            # Store the data
            data_storage_dic[(error_rate, correct_rate)]['error_num'].append(error_num)
            data_storage_dic[(error_rate, correct_rate)]['correct_num'].append(correct_num)

            data_storage_dic[(error_rate, correct_rate)]['expanded_num'].append(expanded_num)
            data_storage_dic[(error_rate, correct_rate)]['planning_time_total'].append(planning_time_total)

            data_storage_dic[(error_rate, correct_rate)]['current_cost'].append(current_cost)
            data_storage_dic[(error_rate, correct_rate)]['act_steps'].append(planning_time_total)
            data_storage_dic[(error_rate, correct_rate)]['bt_status'].append(current_cost)

# Preparing the detailed data for CSV
all_data_records = []
summary_results = []
for key, values in data_storage_dic.items():
    error_rate, correct_rate = key
    for i in range(len(values['expanded_num'])):
        record = {
            'Data ID': i % (len(error_rate_range_ls) * len(correct_rate_range_ls)),
            'Error Rate': error_rate,
            'Correct Rate': correct_rate,
            'Error Num': values['error_num'][i],
            'Correct Num': values['correct_num'][i],
            'Expanded Num': values['expanded_num'][i],
            'Planning Time Total': values['planning_time_total'][i],
            'Current Cost': values['current_cost'][i],
            'Act Steps': values['act_steps'][i],
            'BT Status': values['bt_status'][i]
        }
        all_data_records.append(record)

    error_rate, correct_rate = key
    total_error_num = np.mean(values['error_num'])
    total_correct_num = np.mean(values['correct_num'])
    total_expanded_num = np.mean(values['expanded_num'])
    total_planning_time = np.mean(values['planning_time_total'])
    total_current_cost = np.mean(values['current_cost'])
    total_act_steps = np.mean(values['act_steps'])
    total_bt_status = np.mean(values['bt_status'])

    summary_results.append([
        error_rate, correct_rate, total_error_num, total_correct_num, total_expanded_num,
        total_planning_time, total_current_cost, total_act_steps, total_bt_status
    ])

# Creating DataFrame for detailed and summary data
df_details = pd.DataFrame(all_data_records, columns=[
    'Data ID',
    'Error Rate', 'Correct Rate', 'Error Num', 'Correct Num', 'Expanded Num',
    'Planning Time Total', 'Current Cost', 'Act Steps', 'BT Status'
])

df_summary = pd.DataFrame(summary_results, columns=[
    'Error Rate', 'Correct Rate', 'Total Error Num', 'Total Correct Num', 'Total Expanded Num',
    'Total Planning Time Total', 'Total Current Cost', 'Total Act Steps', 'Total BT Status'
])

# Save detailed data and summary data to CSV
csv_file_path_details = 'output_detailed_bt_cys_10_bigerror.csv'
csv_file_path_summary = 'output_summary_bt_cys_10_bigerror.csv'
df_details.to_csv(csv_file_path_details, index=False)
df_summary.to_csv(csv_file_path_summary, index=False)

print("Detailed data saved to", csv_file_path_details)
print("Summary data saved to", csv_file_path_summary)

all_end_time = time.time()
total_time = (all_end_time - all_start_time)
print("Total Time:", total_time)

# 输出每条数据的
# 对称   0  0.3  0.5  0.7  1
# 数量
# 扩展节点数 时间
# 每条数据有5条这样的，然后保存为 csv

# 求出平均的一个结果，形成一张表
# 0    正确数 错误数  平均总数 expanded_num  time
# 0.3
# 0.5
# 0.7
# 1

# 分开算情况下
# 错误率占比0，正确从0-1
# 错误率占比0.3，正确从0-1
# 错误率占比0，5正确从0-1
# 得到结果

import time
import re
import os
import random
import numpy as np
import pandas as pd
from itertools import chain
from btgym import BehaviorTree, ExecBehaviorLibrary
import btgym
from btgym.utils import ROOT_PATH
from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from btgym.envs.RobotHow.exec_lib._base.RHAction import VHTAction
from sympy import symbols, Not, Or, And, to_dnf, simplify_logic
from btgym.utils.read_dataset import read_dataset
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process, act_format_records
from btgym.utils.tools import collect_action_nodes,extract_objects
from tools import execute_algorithm


# from btgym.envs.virtualhome.exec_lib._base.VHAction import VHAction
# env = btgym.make("VH-PutMilkInFridge")
# cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
# cur_cond_set |= {f'IsClose({arg})' for arg in VHAction.CanOpenPlaces}
# cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHAction.HasSwitchObjects}
# big_actions = collect_action_nodes(env.behavior_lib)
#
# file_name="VS"
# data_path = f"{ROOT_PATH}/../test/SCENES_EXP/{file_name}.txt"
# output_path = f"{ROOT_PATH}/../test/SCENES_EXP/{file_name}_processed_data.txt"
# output_csv_path = f"{ROOT_PATH}/../test/SCENES_EXP/{file_name}_processed_h=1.csv"
# data1 = read_dataset(data_path)
# len_data = len(data1)
# print(f"导入 {len_data} 条数据")
# print(data1[0])

# # ================== RW ===============
# name = "RW"
# dataset = read_dataset(f"{name}_test_50.txt")
# from btgym.envs.RoboWaiter.exec_lib._base.VHTAction import VHTAction
# env = btgym.make("RWEnv")
# cur_cond_set = env.agents[0].condition_set = {'RobotNear(Bar)','Holding(Nothing)' }
# cur_cond_set |= {f'Exists({arg})' for arg in VHTAction.all_object-{'Coffee', 'Water', 'Dessert'}}
# print(f"共收集到 {len(VHTAction.all_object)} 个物体")


# # ================== RHS ===============
# name = "RHS"
# dataset = read_dataset(f"{name}_test_50.txt")
# from btgym.envs.RobotHow_Small.exec_lib._base.VHTAction import VHTAction
# env = btgym.make("VHT-Small")
# cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
# cur_cond_set |= {f'IsClose({arg})' for arg in VHTAction.CAN_OPEN}
# cur_cond_set |= {f'IsUnplugged({arg})' for arg in VHTAction.HAS_PLUG}
# cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHTAction.HAS_SWITCH}
# big_actions = collect_action_nodes(env.behavior_lib)


# ================== VH ===============
# name = "VH"
# dataset = read_dataset(f"{name}_test_50.txt")
# from btgym.envs.virtualhome.exec_lib._base.VHAction import VHAction
# env = btgym.make("VH-PutMilkInFridge")
# cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
# cur_cond_set |= {f'IsClose({arg})' for arg in VHAction.CanOpenPlaces}
# cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHAction.HasSwitchObjects}
# big_actions = collect_action_nodes(env.behavior_lib)


# ================== RHB ===============
name = "RHB"
dataset = read_dataset(f"{name}_test_50.txt")
from btgym.envs.RobotHow.exec_lib._base.RHAction import VHTAction as RHB
env = btgym.make("VHT-PutMilkInFridge")
cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
cur_cond_set |= {f'IsClose({arg})' for arg in RHB.CAN_OPEN}
cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in RHB.HAS_SWITCH}
cur_cond_set |= {f'IsUnplugged({arg})' for arg in RHB.HAS_PLUG}
big_actions = collect_action_nodes(env.behavior_lib)


# Initialize accumulators for lengths
total_priority_act_length = 0
total_algo_actions_length = 0
num_entries = 0
planning_time_total_all = 0
for n,d in enumerate(dataset):
    goal_str = ' & '.join(d["Goals"])
    act_str = ', '.join(d["Optimal Actions"])

    goal_set = goal_transfer_str(goal_str)
    print("goal_set:", goal_set)
    priority_act_ls = act_str_process(act_str)
    print("priority_act_ls:", priority_act_ls)

    key_predicates = extract_objects(priority_act_ls)
    priority_obj_ls = []
    objects = set()
    pattern = re.compile(r'\((.*?)\)')
    for expr in chain(goal_set[0], priority_act_ls):
        match = pattern.search(expr)
        if match:
            objects.update(match.group(1).split(','))
    priority_obj_ls += list(objects)
    print("priority_obj_ls:", priority_obj_ls)

    algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                          priority_act_ls=priority_act_ls, key_predicates=key_predicates,
                          key_objects=priority_obj_ls,
                          selected_algorithm="opt", mode="small-predicate-objs", #mode=""
                          llm_reflect=False, time_limit=180,
                          heuristic_choice=0)

    expanded_num, planning_time_total, cost, error, act_num, current_cost, record_act_ls = \
        execute_algorithm(algo, goal_set, cur_cond_set)
    time_limit_exceeded = algo.algo.time_limit_exceeded
    success = not error and not time_limit_exceeded

    if time_limit_exceeded:
        RED = "\033[31m"
        RESET = "\033[0m"
        print(f"{RED}- ID: {n}  Goal:{goal_str}  Time Out  -{RESET}")
        planning_time_total=180
    planning_time_total_all += planning_time_total
    print("time:",planning_time_total)

    # Update accumulators
    total_priority_act_length += len(priority_act_ls)
    total_algo_actions_length += len(algo.actions)
    num_entries += 1

# Calculate averages
average_priority_act_length = total_priority_act_length / num_entries
average_algo_actions_length = total_algo_actions_length / num_entries

# Print averages
print(f"最优路径长度: {average_priority_act_length}")
print(f"动作空间大小: {average_algo_actions_length}")
print(f"Planning Time Total: {planning_time_total_all/len(dataset)}")
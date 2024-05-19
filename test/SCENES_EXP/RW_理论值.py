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
from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction
from sympy import symbols, Not, Or, And, to_dnf, simplify_logic
from btgym.utils.read_dataset import read_dataset
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process, act_format_records
from btgym.utils.tools import collect_action_nodes,extract_objects



from btgym.envs.virtualhome.exec_lib._base.VHAction import VHAction
from btgym.envs.robowaiter.exec_lib._base.VHTAction import VHTAction
env = btgym.make("RWEnv")
cur_cond_set = env.agents[0].condition_set = {'RobotNear(Bar)','Holding(Nothing)' }
cur_cond_set |= {f'Exists({arg})' for arg in VHTAction.all_object-{'Coffee', 'Water', 'Dessert'}}

big_actions = collect_action_nodes(env.behavior_lib)

file_name="RW_test_50"
data_path = f"{ROOT_PATH}/../test/SCENES_EXP/{file_name}.txt"
data1 = read_dataset(data_path)
len_data = len(data1)
print(f"导入 {len_data} 条数据")
print(data1[0])

# Initialize accumulators for lengths
total_priority_act_length = 0
total_algo_actions_length = 0
num_entries = 0

for n,d in enumerate(data1):
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
                          selected_algorithm="opt", mode="small-predicate-objs",
                          llm_reflect=False, time_limit=None,
                          heuristic_choice=1)
    # Update accumulators
    total_priority_act_length += len(priority_act_ls)
    total_algo_actions_length += len(algo.actions)
    num_entries += 1

# Calculate averages
average_priority_act_length = total_priority_act_length / num_entries
average_algo_actions_length = total_algo_actions_length / num_entries

# Print averages
print(f"Average length of priority_act_ls: {average_priority_act_length}")
print(f"Average length of algo.actions: {average_algo_actions_length}")

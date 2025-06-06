import copy
import os
import matplotlib.pyplot as plt
from collections import Counter
import random
from btgym.utils import ROOT_PATH
import numpy as np
os.chdir(f'{ROOT_PATH}/../z_experience_results')
import time
import re
import btgym
from btgym.utils.tools import collect_action_nodes
from btgym.utils.read_dataset import read_dataset
from btgym.algos.llm_client.tools import goal_transfer_str
from btgym.algos.bt_autogen.main_interface import BTPlannerInterface
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process, act_format_records
from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction
from btgym.envs.RobotHow_Small.exec_lib._base.OGAction import OGAction
from btgym.envs.RoboWaiter.exec_lib._base.RWAction import RWAction
from btgym.envs.VirtualHome.exec_lib._base.VHAction import VHAction


difficulty= "single" #"single"  #"mix" "multi"
scene = "VH"

# 导入数据
data_path = f"{ROOT_PATH}/../z_experience_results/data/{scene}_{difficulty}_100_processed_data.txt"
data = read_dataset(data_path)


from btgym.envs.VirtualHome.exec_lib._base.VHAction import VHAction
env = btgym.make("VH-PutMilkInFridge")
cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
cur_cond_set |= {f'IsClose({arg})' for arg in VHAction.CanOpenPlaces}
cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHAction.HasSwitchObjects}
big_actions = collect_action_nodes(env.behavior_lib)


# for i, d in enumerate(data[:1]):
for i, d in enumerate(data[1:2]): # data[3:4]
    goal_str = ' & '.join(d["Goals"])
    goal_set = goal_transfer_str(goal_str)
    opt_act = act_str_process(d['Optimal Actions'], already_split=True)

    algo = BTPlannerInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                          priority_act_ls=opt_act, key_predicates=[],
                          key_objects=[],
                          selected_algorithm="weak", mode="big",
                          llm_reflect=False, time_limit=15,
                          heuristic_choice=-1,exp=True,output_just_best=True)

    goal_set = goal_transfer_str(goal_str)

    start_time = time.time()
    algo.process(goal_set)
    end_time = time.time()
    planning_time_total = end_time - start_time

    time_limit_exceeded = algo.algo.time_limit_exceeded

    ptml_string, cost, expanded_num = algo.post_process()
    error, state, act_num, current_cost, record_act_ls = algo.execute_bt(goal_set[0], cur_cond_set, verbose=False)

    print(f"\x1b[32m Goal:{goal_str} \n Executed {act_num} action steps\x1b[0m",
          "\x1b[31mERROR\x1b[0m" if error else "",
          "\x1b[31mTIMEOUT\x1b[0m" if time_limit_exceeded else "")
    print("current_cost:", current_cost, "expanded_num:", expanded_num, "planning_time_total:", planning_time_total)

    # visualization
    file_name = "bfs_output_just_best"
    file_path = f'./{file_name}.btml'
    with open(file_path, 'w') as file:
        file.write(ptml_string)
    # read and execute
    from btgym import BehaviorTree

    bt = BehaviorTree(file_name + ".btml", env.behavior_lib)
    bt.print()
    bt.draw()

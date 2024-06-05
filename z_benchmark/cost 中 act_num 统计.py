import copy
import os
import matplotlib.pyplot as plt
from collections import Counter
import random
from btgym.utils import ROOT_PATH
import pandas as pd
import numpy as np
import time
import Attraction.tools as tools
import re
import btgym
from btgym.utils.tools import collect_action_nodes
from btgym.utils.read_dataset import read_dataset
from btgym.algos.llm_client.tools import goal_transfer_str
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process, act_format_records
from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction
from btgym.envs.RobotHow_Small.exec_lib._base.RHSAction import RHSAction
from btgym.envs.RoboWaiter.exec_lib._base.RWAction import RWAction
from btgym.envs.virtualhome.exec_lib._base.VHAction import VHAction
os.chdir(f'{ROOT_PATH}/../z_benchmark')
from btgym.algos.bt_autogen.tools  import calculate_priority_percentage

def get_algo(d,ld,difficulty, scene, algo_str, max_epoch, data_num, save_csv=False):
    goal_str = ' & '.join(d["Goals"])
    goal_set = goal_transfer_str(goal_str)
    opt_act = act_str_process(d['Optimal Actions'], already_split=True)

    heuristic_choice = -1  # obtea, bfs
    algo_str_complete = algo_str
    if algo_str == "opt_h0": heuristic_choice = 0
    elif algo_str == "opt_h0_llm":heuristic_choice = 0
    elif algo_str == "opt_h1": heuristic_choice = 1
    if algo_str in ['opt_h0', 'opt_h1',"opt_h0_llm"]: algo_str = 'opt'

    priority_opt_act=[]
    # 小空间
    if algo_str_complete == "opt_h0_llm":
        priority_opt_act = act_str_process(ld['Optimal Actions'], already_split=True)
        # print("llm_opt_act:",priority_opt_act)
        # print("opt_act:", opt_act)
    elif "opt" in algo_str_complete:
        priority_opt_act=opt_act
    print("opt_act",opt_act)

    algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                          priority_act_ls=priority_opt_act, key_predicates=[],
                          key_objects=[],
                          selected_algorithm=algo_str, mode="big",
                          llm_reflect=False, time_limit=None,
                          heuristic_choice=heuristic_choice,exp=False,exp_cost=True,output_just_best=False,
                          theory_priority_act_ls=opt_act,max_expanded_num=max_epoch)

    start_time = time.time()
    algo.process(goal_set)
    end_time = time.time()

    ### Output
    planning_time_total = end_time - start_time
    time_limit_exceeded = algo.algo.time_limit_exceeded
    ptml_string, cost, expanded_num = algo.post_process(ptml_string=False)
    error, state, act_num, current_cost, record_act_ls,current_tick_time = algo.execute_bt(goal_set[0], cur_cond_set, verbose=False)

    # print("data:", i, "scene:",scene, "algo:",algo_str_complete)
    print(f"\x1b[32m Goal:{goal_str} \n Executed {act_num} action steps\x1b[0m",
          "\x1b[31mERROR\x1b[0m" if error else "",
          "\x1b[31mTIMEOUT\x1b[0m" if time_limit_exceeded else "")
    print("current_cost:", current_cost, "expanded_num:", expanded_num, "planning_time_total:", planning_time_total)

    return algo



max_epoch = 1
data_num = 1
algo_type = ['opt_h0','opt_h0_llm', 'obtea', 'bfs']   # 'opt_h0','opt_h0_llm', 'obtea', 'bfs',      'opt_h1','weak'

algo_act_num_ls = {
    'opt_h0': [],
    'opt_h0_llm': [],
    'obtea': [],
    'bfs': []
}

for difficulty in ['single']:  # 'single', 'multi'
    for scene in ['RH']:  # 'RH', 'RHS', 'RW', 'VH'

        data_path = f"{ROOT_PATH}/../z_benchmark/data/{scene}_{difficulty}_100_processed_data.txt"
        data = read_dataset(data_path)
        llm_data_path = f"{ROOT_PATH}/../z_benchmark/llm_data/{scene}_{difficulty}_100_llm_data.txt"
        llm_data = read_dataset(llm_data_path)
        env, cur_cond_set = tools.setup_environment(scene)

        for i, (d,ld) in enumerate(zip(data[:data_num],llm_data[:data_num])):
            print("data:", i, "difficulty:",difficulty, "scene:",scene,)
            goal_str = ' & '.join(d["Goals"])
            goal_set = goal_transfer_str(goal_str)


            for algo_str in algo_type:  # "opt_h0", "opt_h1", "obtea", "bfs"
                algo = copy.deepcopy(get_algo(d,ld,difficulty, scene, algo_str, max_epoch, data_num, save_csv=True))

                if algo_str in ['opt_h0','opt_h0_llm', 'obtea']:
                    for c_leaf in algo.algo.expanded:
                        c = c_leaf.content
                        error, state, act_num, current_cost, record_act_ls, current_tick_time = algo.execute_bt(
                            goal_set[0], c, verbose=False)
                        algo_act_num_ls[algo_str].append(act_num)
                        # print(algo_str,c, act_num)
                else:
                    for c in algo.algo.traversed:
                        error, state, act_num, current_cost, record_act_ls, current_tick_time = algo.execute_bt(
                            goal_set[0], c, verbose=False)
                        algo_act_num_ls[algo_str].append(act_num)
                        # print(algo_str,c, act_num)


    print(algo_act_num_ls)




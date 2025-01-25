import copy
import os
import matplotlib.pyplot as plt
from collections import Counter
import random
from btgym.utils import ROOT_PATH
import pandas as pd
import numpy as np
import time
import exp4_Attraction.tools as tools
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
from btgym.envs.VirtualHome.exec_lib._base.VHAction import VHAction
os.chdir(f'{ROOT_PATH}/../z_benchmark')
from btgym.algos.bt_autogen.tools  import calculate_priority_percentage

from hbtp.algos.bt_planning.HBTP import HBTP
from hbtp.algos.bt_planning.main_interface import BTExpInterface as HBTPExpInterface

def plot_percentage(percentages_type, difficulty, scene, algo_type, max_epoch, data_num, save_csv=False):
    data_path = f"{ROOT_PATH}/../z_benchmark/data/{scene}_{difficulty}_100_processed_data.txt"
    data = read_dataset(data_path)
    llm_data_path = f"{ROOT_PATH}/../z_benchmark/llm_data/{scene}_{difficulty}_100_llm_data.txt"
    llm_data = read_dataset(llm_data_path)
    env, cur_cond_set = tools.setup_environment(scene)

    for algo_str in algo_type:  # "opt_h0", "opt_h1", "obtea", "bfs"
        print(f"\n======== Start {algo_str} !! =============")
        corr_ratio_all = []  # 记录每个data的ratio

        heuristic_choice = -1  # obtea, bfs
        algo_str_complete = algo_str
        if algo_str == "opt_h0": heuristic_choice = 0
        elif algo_str == "opt_h0_llm":heuristic_choice = 0
        elif algo_str == "opt_h1": heuristic_choice = 1
        elif algo_str == "opt_h1_llm": heuristic_choice = 1
        if algo_str in ['opt_h0', 'opt_h1',"opt_h0_llm","opt_h1_llm"]: algo_str = 'opt'

        # Recording result details
        detail_rows = []

        for i, (d,ld) in enumerate(zip(data[:data_num],llm_data[:data_num])):
            print("data:", i, "scene:",scene, "algo:",algo_str_complete)
            goal_str = ' & '.join(d["Goals"])
            goal_set = goal_transfer_str(goal_str)
            opt_act = act_str_process(d['Optimal Actions'], already_split=True)

            # 小空间
            # algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
            #                       priority_act_ls=opt_act, key_predicates=[],
            #                       key_objects=d['Vital Objects'],
            #                       selected_algorithm=algo_str, mode="small-objs",
            #                       llm_reflect=False, time_limit=5,
            #                       heuristic_choice=heuristic_choice,exp=True,output_just_best=True)
            # 如果是cost  output_just_best=False，这样才会出现非最优？

            priority_opt_act=[]
            # 小空间
            if algo_str_complete in ["opt_h0_llm","opt_h1_llm"] :
                priority_opt_act = act_str_process(ld['Optimal Actions'], already_split=True)
                # print("llm_opt_act:",priority_opt_act)
                # print("opt_act:", opt_act)
            elif "opt" in algo_str_complete:
                priority_opt_act=opt_act
            print("opt_act",opt_act)
            
            


                # algo = HBTPExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                #                       priority_act_ls=[], key_predicates=[],
                #                       key_objects=[],
                #                       selected_algorithm='hbtp', mode="big",
                #                       act_tree_verbose=False, time_limit=5,
                #                       max_expanded_num=50,
                #                       heuristic_choice=-1, output_just_best=False,
                #                       info_dict=info_dict)
                
            # if scene != 'RH':
            #     time_limit = 0.001
            # else:
            #     time_limit = 0.1
            time_limit = 1

            algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                                priority_act_ls=priority_opt_act, key_predicates=[],
                                key_objects=[],
                                selected_algorithm=algo_str, mode="big",
                                llm_reflect=False, time_limit=time_limit, #5
                                heuristic_choice=heuristic_choice,exp=False,exp_cost=False,output_just_best=False,
                                theory_priority_act_ls=opt_act,max_expanded_num=10000000)

            goal_set = goal_transfer_str(goal_str)
            start_time = time.time()
            algo.process(goal_set)
            algo.algo.bt.print_nodes()
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
            
            if algo_str_complete == 'hbtp' or algo_str_complete == 'bfs':
                Tree_Expanded_Number = algo.algo.traversed_state_num
            else:
                Tree_Expanded_Number = expanded_num

            # 记录每个场景 每个算法 每条数据的详细结果 存入 csv
            new_row = {
                'Goal': goal_str,
                'Optimal_Actions': d['Optimal Actions'],
                'LLM_Optimal_Actions': ld['Optimal Actions'],
                'Vital_Action_Predicates': d['Vital Action Predicates'],
                'Vital_Objects': d['Vital Objects'],
                'Time_Limit_Exceeded': time_limit_exceeded,
                'Error': error,
                # 'Expanded_Number': expanded_num,
                'Tree_Expanded_Number': Tree_Expanded_Number, # 实际在树上的节点数
                'Traversed_Number': algo.algo.traversed_state_num,
                'Exploration_Number': expanded_num, # 原来的 Expanded_Number
                'Planning_Time_Total': planning_time_total,
                'Current_Cost': current_cost,
                'Action_Number': act_num,
                'Recorded_Action_List': record_act_ls,
                'Tick_Time':current_tick_time
            }
            detail_rows.append(new_row)

        # save detail to csv
        detailed_df = pd.DataFrame.from_records(detail_rows)
        save_path = f'{ROOT_PATH}/../z_benchmark/algo_details_t=1/{difficulty}_{scene}_{algo_str_complete}.csv'
        detailed_df.to_csv(save_path, index=False)


max_epoch = 2000
data_num = 100
# algo_type = ['hbtp']   # 'opt_h0','opt_h0_llm', 'obtea', 'bfs',      'opt_h1','weak'  #,'opt_h1_llm', 'weak'
algo_type = ['hbtp']

for percentages_type in ['expanded']:  # 'expanded', 'traversed', 'cost'
    for difficulty in ['single', 'multi']:  # 'single', 'multi'
        print(f"============ percentages_type = {percentages_type}, difficulty = {difficulty} =============")
        for scene in ['RW']:  # 'RH', 'RHS', 'RW', 'VH'
            print(f"++++++++++ scene = {scene} ++++++++++")
            plot_percentage(percentages_type, difficulty, scene, algo_type, max_epoch, data_num, save_csv=True)





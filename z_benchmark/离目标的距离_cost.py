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


def plot_percentage(percentages_type, difficulty, scene, algo_type, max_epoch, data_num, save_csv=False):
    data_path = f"{ROOT_PATH}/../z_benchmark/data/{scene}_{difficulty}_100_processed_data.txt"
    data = read_dataset(data_path)
    llm_data_path = f"{ROOT_PATH}/../z_benchmark/llm_data/{scene}_{difficulty}_100_llm_data.txt"
    llm_data = read_dataset(llm_data_path)
    env, cur_cond_set = tools.setup_environment(scene)

    mean_corr_ratio = []  # 存储5个算法下的mean
    std_corr_ratio = []  # 存储5个算法下的std
    for algo_str in algo_type:  # "opt_h0", "opt_h1", "obtea", "bfs"
        print(f"Start {algo_str}!!")
        corr_ratio_all = []  # 记录每个data的ratio

        heuristic_choice = -1  # obtea, bfs
        algo_str_complete = algo_str
        if algo_str == "opt_h0": heuristic_choice = 0
        elif algo_str == "opt_h0_llm":heuristic_choice = 0
        elif algo_str == "opt_h1": heuristic_choice = 1
        if algo_str in ['opt_h0', 'opt_h1',"opt_h0_llm"]: algo_str = 'opt'

        # Initialize DataFrame for recording results
        detailed_results = pd.DataFrame(columns=[
            'Scene', 'Algorithm', 'Data Index', 'Goal String', 'Optimal Actions', 'LLM Optimal Actions',
            'Vital Action Predicates', 'Vital Objects', 'Time Limit Exceeded', 'Error', 'Expanded Number',
            'Planning Time Total', 'Current Cost', 'Action Number', 'Recorded Action List'
        ])

        for i, (d,ld) in enumerate(zip(data[:data_num],llm_data[:data_num])):
            print("i:",i)
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

            # 小空间
            if algo_str_complete == "opt_h0_llm":
                llm_opt_act = act_str_process(ld['Optimal Actions'], already_split=True)
                algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                                      priority_act_ls=llm_opt_act, key_predicates=d['Vital Action Predicates'],
                                      key_objects=d['Vital Objects'],
                                      selected_algorithm=algo_str, mode="small-predicate-objs",
                                      llm_reflect=False, time_limit=5,
                                      heuristic_choice=heuristic_choice,exp=True,exp_cost=False,output_just_best=True)
            # 大空间
            else:
                algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                                      priority_act_ls=opt_act, key_predicates=[],
                                      key_objects=[],
                                      selected_algorithm=algo_str, mode="big",
                                      llm_reflect=False, time_limit=5,
                                      heuristic_choice=heuristic_choice,exp=True,exp_cost=False,output_just_best=True)


            goal_set = goal_transfer_str(goal_str)
            start_time = time.time()
            algo.process(goal_set)
            end_time = time.time()

            ### Output
            planning_time_total = end_time - start_time
            time_limit_exceeded = algo.algo.time_limit_exceeded
            ptml_string, cost, expanded_num = algo.post_process()
            error, state, act_num, current_cost, record_act_ls = algo.execute_bt(goal_set[0], cur_cond_set, verbose=False)
            print(f"\x1b[32m Goal:{goal_str} \n Executed {act_num} action steps\x1b[0m",
                  "\x1b[31mERROR\x1b[0m" if error else "",
                  "\x1b[31mTIMEOUT\x1b[0m" if time_limit_exceeded else "")
            print("current_cost:", current_cost, "expanded_num:", expanded_num, "planning_time_total:", planning_time_total)

            # 记录每个场景 每个算法 每条数据的详细结果 存入 csv
            # goal_str    d['Optimal Actions']   ld['Optimal Actions']  d['Vital Action Predicates']   d['Vital Objects']
            # time_limit_exceeded,  error,  expanded_num, planning_time_total, current_cost, act_num, record_act_ls



            if percentages_type == 'expanded':
                corr_ratio = algo.algo.expanded_percentages
            elif percentages_type == 'traversed':
                corr_ratio = algo.algo.traversed_percentages
            elif percentages_type == 'cost':
                # corr_ratio = algo.algo.max_min_cost_ls
                corr_ratio = algo.algo.simu_cost_ls
            if len(corr_ratio) < max_epoch:
                corr_ratio.extend([corr_ratio[-1]] * (max_epoch - len(corr_ratio)))
            else:
                corr_ratio = corr_ratio[:max_epoch]

            corr_ratio_all.append(corr_ratio)

        # 保存所有epoch的数据
        if save_csv == True:
            if heuristic_choice == 0: algo_str = 'opt_h0'
            if heuristic_choice == 1: algo_str = 'opt_h1'
            df = pd.DataFrame(corr_ratio_all)
            file_path = f'./output_percentage/{percentages_type}_{difficulty}_{scene}_{algo_str}.csv'
            df.to_csv(file_path, index=False, header=False)


        # 可以选择导入corr_ratio_all
        corr_ratio_all = np.array(corr_ratio_all)   # 每个data的所有epoch的ratio
        mean_corr_ratio.append(list(np.mean(corr_ratio_all, axis=0)))  # epoch的平均mean
        std_corr_ratio.append(list(np.std(corr_ratio_all, axis=0)))  # epoch的平均std

    mean_corr_ratio = np.array(mean_corr_ratio)
    std_corr_ratio = np.array(std_corr_ratio)
    epochs = np.arange(1, max_epoch + 1)

    plt.figure(figsize=(10, 6))
    for i, algo_str in enumerate(algo_type):
        # # 误差线
        # # plt.errorbar(epochs, mean_corr_ratio, yerr=std_corr_ratio, fmt='-o', capsize=5, label='Mean with Std Dev')
        # # 误差范围
        plt.plot(epochs, mean_corr_ratio[i], label=f'{algo_str}') # color=color[i],
        plt.fill_between(epochs, mean_corr_ratio[i] - std_corr_ratio[i], mean_corr_ratio[i] + std_corr_ratio[i],
                         alpha=0.2)  # , label=f'{algo_str} Std Dev', color=color[i],

    plt.xlabel('Epoch')
    plt.ylabel(f'{percentages_type} ratio')
    plt.title(f'{percentages_type} ratio in {scene} ({difficulty})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./images_percentage/{percentages_type}_{difficulty}_{scene}.png', dpi=100)
    plt.show()

max_epoch = 1000
data_num = 100
algo_type = ['opt_h0','opt_h0_llm', 'obtea', 'bfs']   # 'opt_h0','opt_h0_llm', 'obtea', 'bfs',      'opt_h1','weak'

for percentages_type in ['expanded']:  # 'expanded', 'traversed', 'cost'
    for difficulty in ['multi']:  # 'single', 'multi'
        print(f"============ percentages_type = {percentages_type}, difficulty = {difficulty} =============")
        for scene in ['RH', 'RHS','VH']:  # 'RH', 'RHS', 'RW', 'VH'
            print(f"++++++++++ scene = {scene} ++++++++++")
            plot_percentage(percentages_type, difficulty, scene, algo_type, max_epoch, data_num, save_csv=True)





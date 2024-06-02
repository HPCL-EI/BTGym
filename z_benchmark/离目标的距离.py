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
    env, cur_cond_set = tools.setup_environment(scene)

    mean_corr_ratio = []  # 存储5个算法下的mean
    std_corr_ratio = []  # 存储5个算法下的std
    for algo_str in algo_type:  # "opt_h0", "opt_h1", "obtea", "bfs"
        print(f"Start {algo_str}!!")
        corr_ratio_all = []  # 记录每个data的ratio

        heuristic_choice = -1  # obtea, bfs
        if algo_str == "opt_h0": heuristic_choice = 0
        elif algo_str == "opt_h1": heuristic_choice = 1
        if algo_str in ['opt_h0', 'opt_h1']: algo_str = 'opt'

        for i, d in enumerate(data[:data_num]):
            goal_str = ' & '.join(d["Goals"])
            goal_set = goal_transfer_str(goal_str)
            opt_act = act_str_process(d['Optimal Actions'], already_split=True)
            algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                                  priority_act_ls=opt_act, key_predicates=[],
                                  key_objects=[],
                                  selected_algorithm=algo_str, mode="big",
                                  llm_reflect=False, time_limit=3,
                                  heuristic_choice=heuristic_choice,exp=True,output_just_best=True)

            goal_set = goal_transfer_str(goal_str)
            # start_time = time.time()
            algo.process(goal_set)
            # end_time = time.time()

            ### Output
            # planning_time_total = end_time - start_time
            # time_limit_exceeded = algo.algo.time_limit_exceeded
            # ptml_string, cost, expanded_num = algo.post_process()
            # error, state, act_num, current_cost, record_act_ls = algo.execute_bt(goal_set[0], cur_cond_set, verbose=False)
            # print(f"\x1b[32m Goal:{goal_str} \n Executed {act_num} action steps\x1b[0m",
            #       "\x1b[31mERROR\x1b[0m" if error else "",
            #       "\x1b[31mTIMEOUT\x1b[0m" if time_limit_exceeded else "")
            # print("current_cost:", current_cost, "expanded_num:", expanded_num, "planning_time_total:", planning_time_total)
            if percentages_type == 'expanded':
                corr_ratio = algo.algo.expanded_percentages
            elif percentages_type == 'traversed':
                corr_ratio = algo.algo.traversed_percentages
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
            file_path = f'./output_percentage/{percentages_type}_{scene}_{difficulty}_{algo_str}.csv'
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
    plt.ylabel('Mean Corr Ratio')
    plt.title(f'Mean Corr Ratio in {scene}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./images_percentage/{percentages_type}_{difficulty}_{scene}.png', dpi=100)
    plt.show()

max_epoch = 20
data_num = 5
algo_type = ['opt_h0', 'opt_h1', 'obtea', 'bfs']   # 'opt_h0', 'opt_h1', 'obtea', 'bfs'

for percentages_type in ['expanded']:  # 'expanded', 'traversed'
    for difficulty in ['multi']:  # 'single', 'multi'
        print(f"============ percentages_type = {percentages_type}, difficulty = {difficulty} =============")
        for scene in ['RHS']:  # 'RH', 'RHS', 'RW', 'VH'
            print(f"++++++++++ scene = {scene} ++++++++++")
            plot_percentage(percentages_type, difficulty, scene, algo_type, max_epoch, data_num, save_csv=True)





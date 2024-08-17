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
from btgym.envs.VirtualHome.exec_lib._base.VHAction import VHAction
os.chdir(f'{ROOT_PATH}/../z_benchmark')
from btgym.algos.bt_autogen.tools  import calculate_priority_percentage

def get_algo(d,ld,difficulty, scene, algo_str, max_epoch, save_csv=False):
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


import pandas as pd
import matplotlib.pyplot as plt



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# def plot_adaptive_histograms_and_save(difficulty, scene):
#     algo_names = ['opt_h0', 'opt_h0_llm', 'bfs']
#     data_frames = []
#
#     # 读取所有算法的CSV文件并存储在data_frames列表中
#     for algo_name in algo_names:
#         filename = f"./COST_output/{difficulty}_{scene}_{algo_name}.csv"
#         df = pd.read_csv(filename)
#         df['algo_name'] = algo_name
#         df['cost_ratio'] = df['algo_cost'] /df['obtea_cost']
#         data_frames.append(df)
#
#     # 合并所有数据
#     combined_df = pd.concat(data_frames)
#
#     # 计算自适应bins
#     min_cost = combined_df['obtea_cost'].min()
#     max_cost = combined_df['obtea_cost'].max()
#     bins = np.arange(min_cost, max_cost + 10, 10)
#
#     # 定义颜色
#     colors = {
#         'opt_h0': "#1f77b4", # 'blue',
#         'opt_h0_llm': '#2ca02c', # orange
#         'bfs': '#ff7f0e' #green
#     }
#
#     plt.figure(figsize=(12, 8))
#
#     # 存储每个类别下三种算法的平均数据
#     average_data = []
#
#     for algo_name in algo_names:
#         algo_df = combined_df[combined_df['algo_name'] == algo_name].copy()
#         algo_df.loc[:, 'group'] = pd.cut(algo_df['obtea_cost'], bins=bins, right=False)
#
#         # 按组计算平均成本比例
#         grouped = algo_df.groupby('group', observed=True)['cost_ratio'].mean()
#
#         if not grouped.empty:
#             grouped.plot(kind='bar',  label=algo_name,color=colors[algo_name]) # ,color=colors[algo_name]
#             # 添加平均数据到列表
#             for group, value in grouped.items():
#                 average_data.append((group, algo_name, value))
#
#     plt.title(f'Combined Cost Ratio Histograms for {scene} in {difficulty}')
#     plt.xlabel('Cost Group')
#     plt.ylabel('Average Cost Ratio')
#     y_min = 0
#     y_max = 2
#     plt.ylim(y_min, y_max)  # 设置y轴范围
#     plt.legend(title='Algorithm')
#
#     # 创建输出目录（如果不存在）
#     output_dir = "./COST_Histograms"
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     # 保存图片到文件
#     output_filename = f"{output_dir}/{difficulty}_{scene}_combined.png"
#     plt.savefig(output_filename)
#     plt.show()  # 显示图表
#     plt.close()  # 关闭图表以释放内存
#     print(f"直方图已保存为：{output_filename}")
#
#     # 输出每个类别下三种算法的平均数据
#     print("\n每个类别下三种算法的平均数据：")
#     for group, algo_name, value in average_data:
#         print(f"类别: {group}, 算法: {algo_name}, 平均成本比例: {value:.2f}")

def plot_adaptive_histograms_and_save(difficulty, scene):
    algo_names = ['opt_h0', 'opt_h0_llm', 'bfs']
    data_frames = []

    # 读取所有算法的CSV文件并存储在data_frames列表中
    for algo_name in algo_names:
        filename = f"./COST_output/{difficulty}_{scene}_{algo_name}.csv"
        df = pd.read_csv(filename)
        df['algo_name'] = algo_name
        df['cost_ratio'] = df['algo_cost'] / df['obtea_cost']
        data_frames.append(df)

    # 合并所有数据
    combined_df = pd.concat(data_frames)

    # 计算自适应bins
    min_cost = combined_df['obtea_cost'].min()
    max_cost = combined_df['obtea_cost'].max()
    bins = np.arange(min_cost, max_cost + 10, 10)

    # 定义颜色和位置偏移
    colors = {
        'opt_h0': '#1f77b4',  # 蓝色
        'opt_h0_llm': '#2ca02c',  # 绿色
        'bfs': '#ff7f0e'  # 橙色
    }
    offsets = {
        'opt_h0': -0.2,
        'opt_h0_llm': 0.0,
        'bfs': 0.2
    }

    plt.figure(figsize=(12, 8))

    # 存储每个类别下三种算法的平均数据
    average_data = []

    for algo_name in algo_names:
        algo_df = combined_df[combined_df['algo_name'] == algo_name].copy()
        algo_df.loc[:, 'group'] = pd.cut(algo_df['obtea_cost'], bins=bins, right=False)

        # 按组计算平均成本比例
        grouped = algo_df.groupby('group', observed=True)['cost_ratio'].mean().dropna()

        if not grouped.empty:
            x = np.arange(len(grouped.index)) + offsets[algo_name]
            plt.bar(x, grouped, width=0.4, color=colors[algo_name], label=algo_name, align='center')
            # 添加平均数据到列表
            for group, value in grouped.items():
                average_data.append((group, algo_name, value))

    plt.title(f'Combined Cost Ratio Histograms for {scene} in {difficulty}')
    plt.xlabel('Cost Group')
    plt.ylabel('Average Cost Ratio')
    y_min = 0
    y_max = 2
    plt.ylim(y_min, y_max)  # 设置y轴范围
    plt.xticks(np.arange(len(grouped.index)), [str(g) for g in grouped.index], rotation=45)
    plt.legend(title='Algorithm')

    # 创建输出目录（如果不存在）
    output_dir = "./COST_Histograms"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存图片到文件
    output_filename = f"{output_dir}/{difficulty}_{scene}_combined.png"
    plt.savefig(output_filename)
    plt.show()  # 显示图表
    plt.close()  # 关闭图表以释放内存
    print(f"直方图已保存为：{output_filename}")

    # 输出每个类别下三种算法的平均数据
    print("\n每个类别下三种算法的平均数据：")
    for group, algo_name, value in average_data:
        print(f"类别: {group}, 算法: {algo_name}, 平均成本比例: {value:.2f}")




max_epoch_obtea = 300
max_epoch = 100
data_num = 100
algo_type = ['opt_h0','opt_h0_llm','bfs','obtea']   # 'opt_h0','opt_h0_llm', 'obtea', 'bfs',      'opt_h1','weak'
algo_dic = {}

# algo_act_num_ls = {
#     'opt_h0': [],
#     'opt_h0_llm': [],
#     'obtea': [],
#     'bfs': []
# }

for difficulty in ['multi']:  # 'single', 'multi'
    for scene in ['RH']:  # 'RH', 'RHS', 'RW', 'VH'

        algo_cost_ls = {
            # 本算法和optea
            'opt_h0': [[], []],
            'opt_h0_llm': [[], []],
            'bfs': [[], []]
        }

        data_path = f"{ROOT_PATH}/../z_benchmark/data/{scene}_{difficulty}_100_processed_data.txt"
        data = read_dataset(data_path)
        llm_data_path = f"{ROOT_PATH}/../z_benchmark/llm_data/{scene}_{difficulty}_100_llm_data.txt"
        llm_data = read_dataset(llm_data_path)
        env, cur_cond_set = tools.setup_environment(scene)

        for i, (d,ld) in enumerate(zip(data[:data_num],llm_data[:data_num])):
            print("data:", i, "difficulty:",difficulty, "scene:",scene,)
            goal_str = ' & '.join(d["Goals"])
            goal_set = goal_transfer_str(goal_str)

            algo_dic["obtea"] = copy.deepcopy(get_algo(d, ld, difficulty, scene, 'obtea', max_epoch=max_epoch_obtea, save_csv=True))


            # 跑每个算法测试
            for algo_str in ["opt_h0", "opt_h0_llm", "bfs"]:
                algo_dic[algo_str] = copy.deepcopy(get_algo(d,ld,difficulty, scene, algo_str, max_epoch, save_csv=True))

                if algo_str in ['opt_h0','opt_h0_llm']:
                    for c_leaf in algo_dic[algo_str].algo.expanded:
                        c = c_leaf.content

                        # obtea 先跑
                        obtea_error, _, act_num, obtea_cost, _, _ = algo_dic["obtea"].execute_bt(
                            goal_set[0], c, verbose=False)
                        if not obtea_error and obtea_cost>0:
                            error, _, act_num, current_cost, _, _ = algo_dic[algo_str].execute_bt(
                                goal_set[0], c, verbose=False)

                            algo_cost_ls[algo_str][0].append(current_cost)
                            algo_cost_ls[algo_str][1].append(obtea_cost)


                        # algo_act_num_ls[algo_str].append(act_num)
                        # print(algo_str,c, act_num)
                else:
                    for c in algo_dic[algo_str].algo.traversed:

                        # obtea 先跑
                        obtea_error, _, act_num, obtea_cost, _, _ = algo_dic["obtea"].execute_bt(
                            goal_set[0], c, verbose=False)
                        if not obtea_error and obtea_cost>0:
                            error, _, act_num, current_cost, _, _ = algo_dic[algo_str].execute_bt(
                                goal_set[0], c, verbose=False)

                            algo_cost_ls[algo_str][0].append(current_cost)
                            algo_cost_ls[algo_str][1].append(obtea_cost)

        # 遍历数据并保存到CSV文件
        for algo_name, costs in algo_cost_ls.items():
            # 创建一个DataFrame
            df = pd.DataFrame({
                'algo_cost': costs[0],
                'obtea_cost': costs[1]
            })
            # 设置CSV文件名
            filename = f"./COST_output/{difficulty}_{scene}_{algo_name}.csv"
            # 保存到CSV
            df.to_csv(filename, index=False)
            print(f"数据已保存到 {filename}")

        # 读入数据跑统计直方图
        # 写成一个函数
        plot_adaptive_histograms_and_save(difficulty, scene)



    # print(algo_act_num_ls)
# 保存数据
# 遍历字典，key 是算法的名字，value 是与该算法相关联的列表
for key, value in algo_cost_ls.items():
    print("算法名称:", key)
    print("相关数据:", value)

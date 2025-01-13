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



max_epoch = 300
data_num = 100
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
    with open(f'algo_act_num_ls_{difficulty}_{scene}.txt', 'w') as file:
        # 写入数据
        file.write(str(algo_act_num_ls))

    # import matplotlib.pyplot as plt
    # from collections import Counter
    # import numpy as np
    #
    # A = algo_act_num_ls['opt_h0']
    # B = algo_act_num_ls['opt_h0_llm']
    # C = algo_act_num_ls['obtea']
    # D = algo_act_num_ls['bfs']
    #
    # # 使用Counter统计频次
    # counts1 = Counter(A)
    # counts2 = Counter(B)
    # counts3 = Counter(C)
    # counts4 = Counter(D)
    #
    # # 合并四个Counter对象，以便我们可以获得完整的x轴范围
    # all_counts = counts1 + counts2 + counts3 + counts4
    # sorted_x = sorted(all_counts.keys())
    #
    #
    # # 初始化y值为0，用于累加柱状图高度（如果需要重叠）
    # y1 = [counts1[x] if x in counts1 else 0 for x in sorted_x]
    # y2 = [counts2[x] if x in counts2 else 0 for x in sorted_x]
    # y3 = [counts3[x] if x in counts3 else 0 for x in sorted_x]
    # y4 = [counts4[x] if x in counts4 else 0 for x in sorted_x]
    #
    # # 绘制第一个柱状图
    #
    # # BTExpansion
    # plt.bar(sorted_x, y4, width=0.8, color='green', edgecolor='black', label='BT Expansion', alpha=0.1)
    # poly = np.polyfit(sorted_x, y4, deg=2)
    # y_value = np.polyval(poly, sorted_x)
    # plt.plot(sorted_x, y_value, color='green', alpha=0.1)
    #
    #
    # # OBTEA
    # plt.bar(sorted_x, y3, width=0.8, color='lightblue', edgecolor='black', label='OBTEA', alpha=0.2)
    # poly = np.polyfit(sorted_x, y3, deg=2)
    # y_value = np.polyval(poly, sorted_x)
    # plt.plot(sorted_x, y_value, color='lightblue', alpha=0.2)
    #
    # # HOBTEA
    # # 绘制第二个柱状图（相邻但不重叠）
    # # 注意：如果你想要重叠，可以将bottom参数设置为y1，但通常不推荐这样做
    # plt.bar(sorted_x, y2, width=0.8, color='orange', edgecolor='black', label='HOBTEA', alpha=0.1)
    # poly = np.polyfit(sorted_x, y2, deg=2)
    # y_value = np.polyval(poly, sorted_x)
    # plt.plot(sorted_x, y_value, color='orange', alpha=0.1)
    #
    # # HOBTEA - Oracle
    # plt.bar(sorted_x, y1, width=0.8, color='red', edgecolor='black', label='HOBTEA-Oracle', alpha=0.1)
    # # 绘制曲线
    # poly = np.polyfit(sorted_x, y1, deg=2)
    # y_value = np.polyval(poly, sorted_x)
    # plt.plot(sorted_x, y_value, color='red', alpha=0.1)
    #
    #
    # # 添加标题和轴标签
    # plt.title('Frequency of Values in Two Lists')
    # plt.xlabel('Values')
    # plt.ylabel('Frequency')
    #
    # # 添加图例
    # plt.legend()
    # plt.subplots_adjust(bottom=0.3)
    # plt.savefig(f'./算法效率的对比图/com.png',
    #             dpi=100, bbox_inches='tight')
    # # 显示图形
    # plt.show()





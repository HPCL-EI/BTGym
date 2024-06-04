import copy
import os
import random
from btgym.utils import ROOT_PATH
os.chdir(f'{ROOT_PATH}/../z_benchmark')
from tools import *
import time
import re
import pandas as pd
import btgym
from btgym.utils.tools import collect_action_nodes
from btgym.utils.read_dataset import read_dataset
from btgym.algos.llm_client.tools import goal_transfer_str
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process, act_format_records

from btgym.envs.RoboWaiter.exec_lib._base.RWAction import RWAction
from btgym.envs.virtualhome.exec_lib._base.VHAction import VHAction
from btgym.envs.RobotHow_Small.exec_lib._base.RHSAction import RHSAction
from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction

SENCE_ACT_DIC={"RW":RWAction,
               "VH":VHAction,
               "RHS":RHSAction,
               "RH":RHAction}

def get_SR(scene, algo_str, just_best,exe_times=5,data_num=100):

    AVG_SR = 0

    # 导入数据
    data_path = f"{ROOT_PATH}/../z_benchmark/data/{scene}_single_100_processed_data.txt"
    data = read_dataset(data_path)
    llm_data_path = f"{ROOT_PATH}/../z_benchmark/llm_data/{scene}_single_100_llm_data.txt"
    llm_data = read_dataset(llm_data_path)
    env, cur_cond_set = setup_environment(scene)

    for i, (d, ld) in enumerate(zip(data[:data_num], llm_data[:data_num])):
        print("data::", i)
        goal_str = ' & '.join(d["Goals"])
        goal_set = goal_transfer_str(goal_str)
        opt_act = act_str_process(d['Optimal Actions'], already_split=True)

        algo_str_complete = algo_str
        heuristic_choice=-1
        if algo_str == "opt_h0" or algo_str == "opt_h0_llm": heuristic_choice = 0
        elif algo_str == "opt_h1": heuristic_choice = 1
        if algo_str in ['opt_h0', 'opt_h1',"opt_h0_llm"]: algo_str = 'opt'

        priority_opt_act = []
        # 小空间
        if algo_str_complete == "opt_h0_llm":
            priority_opt_act = act_str_process(ld['Optimal Actions'], already_split=True)
            print("llm_opt_act:", priority_opt_act)
            print("opt_act:", opt_act)
        elif "opt" in algo_str_complete:
            priority_opt_act = opt_act

        algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                              priority_act_ls=priority_opt_act, key_predicates=[],
                              key_objects=[],
                              selected_algorithm=algo_str, mode="big",
                              llm_reflect=False, time_limit=5,
                              heuristic_choice=heuristic_choice, exp=True, exp_cost=False, output_just_best=True,
                              theory_priority_act_ls=opt_act)

        # 跑算法
        # 提取出obj
        objects = []
        pattern = re.compile(r'\((.*?)\)')
        for expr in goal_set[0]:
            match = pattern.search(expr)
            if match:
                objects.append(match.group(1).split(','))
        successful_executions = 0  # 用于跟踪成功（非错误）的执行次数
        # 随机生成exe_times个初始状态，看哪个能达到目标
        for i in range(exe_times):
            print("----------")
            print("i:", i)
            new_cur_state = modify_condition_set(scene,SENCE_ACT_DIC[scene], cur_cond_set)
            error, state, act_num, current_cost, record_act_ls, ticks = algo.execute_bt(goal_set[0], new_cur_state,
                                                                                        verbose=False)
            # 检查是否有错误，如果没有，则增加成功计数
            if not error:
                successful_executions += 1
            print("----------")
        # 计算非错误的执行占比
        success_ratio = successful_executions / exe_times
        AVG_SR += success_ratio

    AVG_SR = AVG_SR / data_num
    print("成功的执行占比（非错误）: {:.2%}".format(AVG_SR))
    return round(AVG_SR, 2)

algorithms = ['opt_h0','opt_h0_llm', 'obtea', 'bfs']  # 'opt_h0', 'opt_h1', 'obtea', 'bfs', 'dfs'
scenes = ['RH', 'RHS', 'VH']  # 'RH', 'RHS', 'RW', 'VH'
just_best_bts = [True, False] # True, False


data_num=1

# 创建df
index = [f'{algo_str}_{tb}' for tb in ['T', 'F'] for algo_str in algorithms ]
df = pd.DataFrame(index=index, columns=scenes)
for just_best in just_best_bts:
    for algo_str in algorithms:
        index_key = f'{algo_str}_{"T" if just_best else "F"}'
        for scene in scenes:
            df.at[index_key, scene] = get_SR(scene, algo_str, just_best,exe_times=5,data_num=data_num)

formatted_string = df.to_csv(sep='\t')
print(formatted_string)
print("----------------------")
print(df)
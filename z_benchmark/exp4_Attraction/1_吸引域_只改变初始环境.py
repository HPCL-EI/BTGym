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
from btgym.envs.VirtualHome.exec_lib._base.VHAction import VHAction
from btgym.envs.RobotHow_Small.exec_lib._base.RHSAction import RHSAction
from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction
import concurrent.futures

SENCE_ACT_DIC={"RW":RWAction,
               "VH":VHAction,
               "RHS":RHSAction,
               "RH":RHAction}

def get_SR(scene, algo_str, just_best,exe_times=5,data_num=100,difficulty="multi", choose_max_exp=False, max_exp=99999):

    AVG_SR = 0

    # 导入数据
    data_path = f"{ROOT_PATH}/../z_benchmark/data/{scene}_{difficulty}_100_processed_data.txt"
    data = read_dataset(data_path)
    llm_data_path = f"{ROOT_PATH}/../z_benchmark/llm_data/{scene}_{difficulty}_100_llm_data.txt"
    llm_data = read_dataset(llm_data_path)
    env, cur_cond_set = setup_environment(scene)

    algo_str_complete = algo_str
    heuristic_choice = -1
    if algo_str == "opt_h0" or algo_str == "opt_h0_llm":
        heuristic_choice = 0
    elif algo_str == "opt_h1":
        heuristic_choice = 1
    if algo_str in ['opt_h0', 'opt_h1', "opt_h0_llm"]: algo_str = 'opt'


    for i, (d, ld) in enumerate(zip(data[:data_num], llm_data[:data_num])):
        print("data:", i, "scene:",scene, "algo:",algo_str_complete, "just_best:", just_best)
        goal_str = ' & '.join(d["Goals"])
        goal_set = goal_transfer_str(goal_str)
        opt_act = act_str_process(d['Optimal Actions'], already_split=True)

        priority_opt_act = []
        # 小空间
        if algo_str_complete == "opt_h0_llm":
            priority_opt_act = act_str_process(ld['Optimal Actions'], already_split=True)
            # print("llm_opt_act:", priority_opt_act)
            # print("opt_act:", opt_act)
        elif "opt" in algo_str_complete:
            priority_opt_act = opt_act

        if choose_max_exp:
            algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                                  priority_act_ls=priority_opt_act, key_predicates=[],
                                  key_objects=[],
                                  selected_algorithm=algo_str, mode="big",
                                  llm_reflect=False, time_limit=None,
                                  heuristic_choice=heuristic_choice, exp=False, exp_cost=True, output_just_best=just_best,
                                  theory_priority_act_ls=opt_act,max_expanded_num=max_exp)
        else:
            algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                                  priority_act_ls=priority_opt_act, key_predicates=[],
                                  key_objects=[],
                                  selected_algorithm=algo_str, mode="big",
                                  llm_reflect=False, time_limit=5,
                                  heuristic_choice=heuristic_choice, exp=False, exp_cost=False, output_just_best=just_best,
                                  theory_priority_act_ls=opt_act)

        goal_set = goal_transfer_str(goal_str)
        start_time = time.time()
        algo.process(goal_set)
        end_time = time.time()
        planning_time_total = end_time - start_time
        print(f"\x1b[32m Goal:{goal_str} \n Times {planning_time_total} \x1b[0m")

        # time_limit_exceeded = algo.algo.time_limit_exceeded
        # ptml_string, cost, expanded_num = algo.post_process(ptml_string=False)
        # error, state, act_num, current_cost, record_act_ls, ticks = algo.execute_bt(goal_set[0], cur_cond_set,
        #                                                                             verbose=False)
        # print(f"\x1b[32m Goal:{goal_str} \n Executed {act_num} action steps\x1b[0m",
        #       "\x1b[31mERROR\x1b[0m" if error else "",
        #       "\x1b[31mTIMEOUT\x1b[0m" if time_limit_exceeded else "")
        # print("current_cost:", current_cost, "expanded_num:", expanded_num, "planning_time_total:", planning_time_total,
        #       "ticks:", ticks)


        # visualization
        # file_name = "tree"
        # file_path = f'./{file_name}.btml'
        # with open(file_path, 'w') as file:
        #     file.write(ptml_string)
        # # read and execute
        # from btgym import BehaviorTree
        # bt = BehaviorTree(file_name + ".btml", env.behavior_lib)
        # # bt.print()
        # bt.draw()
        pair_num=0
        if algo_str_complete in ['opt_h0','opt_h0_llm', 'obtea']:
            pair_num = len(algo.algo.expanded)
        else:
            pair_num = algo.algo.traversed_state_num

        # 跑算法
        # 提取出obj
        objects = []
        pattern = re.compile(r'\((.*?)\)')
        for expr in goal_set[0]:
            match = pattern.search(expr)
            if match:
                objects.extend(match.group(1).split(','))
        successful_executions = 0  # 用于跟踪成功（非错误）的执行次数
        # 随机生成exe_times个初始状态，看哪个能达到目标
        for i in range(exe_times):
            new_cur_state = modify_condition_set(scene,SENCE_ACT_DIC[scene], cur_cond_set,objects)
            # print("new_cur_state:",new_cur_state)
            error, state, act_num, current_cost, record_act_ls, ticks = algo.execute_bt(goal_set[0], new_cur_state,
                                                                                        verbose=False)
            # 检查是否有错误，如果没有，则增加成功计数
            if not error:
                successful_executions += 1
        # 计算非错误的执行占比
        success_ratio = successful_executions / exe_times
        AVG_SR += success_ratio

    AVG_SR = AVG_SR / data_num
    print("成功的执行占比（非错误）: {:.2%}".format(AVG_SR))
    return pair_num,round(AVG_SR, 4)

def compute_sr_and_update_df(index_key, scene, just_best, exe_times, data_num, difficulty,choose_max_exp=False, max_exp=99999):
    pair_num, sr = get_SR(scene, algo_str, just_best, exe_times=exe_times, data_num=data_num,difficulty=difficulty,
                          choose_max_exp=choose_max_exp, max_exp=max_exp)
    return index_key, scene, pair_num, sr



algorithms = ['bfs','obtea','opt_h0','hbtp','opt_h0_llm']  # 'opt_h0','opt_h0_llm', 'obtea', 'bfs'
scenes = ['RW', 'VH', 'RHS', 'RH']  # 'RW', 'VH', 'RHS', 'RH'
just_best_bts = [False] # True, False


data_num= 100
exe_times =5
difficulty = "multi"

max_exp = 0
choose_max_exp=False

# 创建df
index = [f'{algo_str}_{tb}' for tb in ['T', 'F'] for algo_str in algorithms ]
# index = [f'{algo_str}_{tb}' for tb in ['F'] for algo_str in algorithms ]
df = pd.DataFrame(index=index, columns= scenes)


# for just_best in just_best_bts:
#     for algo_str in algorithms:
#         index_key = f'{algo_str}_{"T" if just_best else "F"}'
#         for scene in scenes:
#             pair_num, df.at[index_key, scene] = get_SR(scene, algo_str, just_best,exe_times=exe_times,data_num=data_num,difficulty=difficulty)
with concurrent.futures.ThreadPoolExecutor() as executor: #max_workers=10
    future_to_sr = {}
    for just_best in just_best_bts:
        for algo_str in algorithms:
            index_key = f'{algo_str}_{"T" if just_best else "F"}'
            for scene in scenes:
                # 将任务提交到线程池
                future = executor.submit(compute_sr_and_update_df, index_key, scene, just_best, exe_times, data_num,difficulty,
                                         choose_max_exp,max_exp)
                future_to_sr[future] = (index_key, scene)

    # 获取并处理每个future的结果
    for future in concurrent.futures.as_completed(future_to_sr):
        index_key, scene = future_to_sr[future]
        try:
            _, _, pair_num, sr = future.result()
            df.at[index_key, scene] = sr
            # df.at[index_key, f"{scene}_PairNum"] = pair_num
        except Exception as exc:
            print(f"{index_key} generated an exception: {exc}")


formatted_string = df.to_csv(sep='\t')
print(formatted_string)
print("----------------------")
print(df)

# Save the DataFrame to a CSV file
if choose_max_exp:
    csv_file_path = f"{ROOT_PATH}/../z_benchmark/exp4_Attraction/1_only_changes_initial_t={exe_times}_False_{difficulty}_max_exp={max_exp}.csv"  # Define your CSV file path
else:
    csv_file_path = f"{ROOT_PATH}/../z_benchmark/exp4_Attraction/1_only_changes_initial_t={exe_times}_False_{difficulty}.csv"  # Define your CSV file path
df.to_csv(csv_file_path)  # Save as a TSV (Tab-separated values) file

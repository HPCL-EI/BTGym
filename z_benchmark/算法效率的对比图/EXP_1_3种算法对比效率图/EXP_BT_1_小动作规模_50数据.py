import re
import time
import random
import numpy as np
import pandas as pd
import pickle

seed = 0
random.seed(seed)
np.random.seed(seed)
from sympy import symbols, Not, Or, And, to_dnf
from sympy import symbols, simplify_logic
import btgym
from btgym import BehaviorTree
from btgym import ExecBehaviorLibrary
from btgym.utils import ROOT_PATH
from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
from btgym.algos.llm_client.llms.gpt4 import LLMGPT4
from btgym.algos.bt_autogen.main_interface import BTExpInterface, collect_action_nodes
from btgym.envs.RobotHow.exec_lib._base.VHTAction_small import VHTAction_small
from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction
from btgym.algos.bt_autogen.tools import state_transition
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process, act_format_records
from btgym.algos.bt_autogen.Action import Action
from btgym.utils.tools import collect_action_nodes, save_data_txt
from btgym.utils.read_dataset import read_dataset
from btgym.algos.llm_client.llm_ask_tools import extract_llm_from_instr_goal
from tools import count_accuracy, identify_and_print_diffs, analyze_data_tabular,generate_custom_action_list
import matplotlib.pyplot as plt


from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction as RHB
env = btgym.make("VHT-PutMilkInFridge")
cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
cur_cond_set |= {f'IsClose({arg})' for arg in RHB.CAN_OPEN}
cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in RHB.HAS_SWITCH}
cur_cond_set |= {f'IsUnplugged({arg})' for arg in RHB.HAS_PLUG}
big_actions = collect_action_nodes(env.behavior_lib)




# 导入数据
# data_path = f"{ROOT_PATH}/../test/BT_EXP/data_small_100.txt"
data_path = f"./DATA_BT_100_ori_yz_revby_cys.txt"
data1 = read_dataset(data_path)
len_data = len(data1)
print(f"导入 {len_data} 条数据")
print(data1[0])
analyze_data_tabular(data1,[97,1,1,1])


# Initialize lists to store individual results and averages
data_rows = []
# action_space_sizes = list(range(15, 91, 5)) #51
action_space_sizes = list(range(15, 101, 5)) #51
heuristic_choices = [0, 1, -1]
average_rows = []

num_samples = 100  # Update this to the number of samples to process


for sample_idx in range(min(len(data1), num_samples)):

    # 测试其中第1条
    d = data1[sample_idx]
    # goals = "IsClean_bananas & IsCut_bananas & IsIn_bananas_fridge"
    goal_str = ' & '.join(d["Goals"])
    # goal_str="IsOn_pear_kitchentable & IsOn_bananas_kitchentable"
    goal_set = goal_transfer_str(goal_str)
    print("sample_idx:",sample_idx,"goal_set:", goal_set)
    d['Optimal Actions'] = act_str_process(d['Optimal Actions'], already_split=True)
    # print("id:", " ", "goals:", d['Goals'])
    print("Optimal Actions:", d['Optimal Actions'])

    for action_space_size in action_space_sizes:
        # print(f"\n============== action_space_sizes = {action_space_size} =============")
        custom_action_list = generate_custom_action_list(big_actions, action_space_size, d['Optimal Actions'])

        # for heuristic_choice in [0, 1, -1]:
        for heuristic_choice in [-1]:
            # if heuristic_choice == 0:
            #     print("----------priority = 0 的启发式 Running with Heuristic Choice = 0-----------")
            # elif heuristic_choice == 1:
            #     print("--------priority = cost/10000 的启发式 Running with Heuristic Choice = 1---------")
            # else:
            #     print("------------无启发式 Running with Heuristic Choice = -1--------")

            algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set, \
                                  priority_act_ls=d['Optimal Actions'],  \
                                  selected_algorithm="weak", mode="user-defined", action_list=custom_action_list,\
                                  llm_reflect=False, time_limit=None,
                                  heuristic_choice=heuristic_choice)

            start_time = time.time()
            algo.process(goal_set)
            end_time = time.time()

            ptml_string, cost, expanded_num = algo.post_process()
            planning_time_total = (end_time - start_time)
            # print("expanded_num:",expanded_num,"planning_time_total",planning_time_total)

            goal = goal_set[0]
            state = cur_cond_set
            error, state, act_num, current_cost, record_act_ls = algo.execute_bt(goal, state, verbose=False)

            time_limit_exceeded = algo.algo.time_limit_exceeded
            algo.algo.clear()

            # Append individual results
            data_rows.append({
                'Sample_ID': sample_idx,
                'Goal': goal_str,
                'Optimal Actions': d['Optimal Actions'],
                'Action_Space_Size': action_space_size,
                'Heuristic_Choice': heuristic_choice,
                'Expanded_Number': expanded_num,
                'Planning_Time_Total': planning_time_total,
                'Current_Cost': current_cost,
                'Error': error,
                'Time_Limit_Exceeded': time_limit_exceeded
            })

# Convert individual results to a DataFrame
df = pd.DataFrame(data_rows)

# Calculate averages grouped by heuristic choice and action space size
numeric_columns = ['Action_Space_Size', 'Heuristic_Choice', 'Expanded_Number', 'Planning_Time_Total', 'Current_Cost', 'Error']
average_df = df[numeric_columns].groupby(['Action_Space_Size', 'Heuristic_Choice']).mean().reset_index()

# average_df = df.groupby(['Action_Space_Size', 'Heuristic_Choice']).mean().reset_index()

# Save individual and average results to CSV
time_str = time.strftime('%Y%m%d%H%M%S', time.localtime())
df.to_csv(f'EXP_1_individual_results_100_size=90_{time_str}.csv', index=False)
average_df.to_csv(f'EXP_1_average_results_100__size=90_{time_str}.csv', index=False)

from scipy.interpolate import interp1d
plt.figure(figsize=(10, 6))
for heuristic_choice in heuristic_choices:
    subset = average_df[average_df['Heuristic_Choice'] == heuristic_choice]
    if not subset.empty:
        # 使用样条插值平滑数据
        x = subset['Action_Space_Size']
        y = subset['Planning_Time_Total']
        spline = interp1d(x, y, kind='cubic')  # 创建一个三次样条插值函数
        xnew = np.linspace(x.min(), x.max(), 300)  # 生成新的x值，用于绘制平滑曲线
        ynew = spline(xnew)  # 生成平滑的y值
        plt.plot(xnew, ynew, label=f'Heuristic Choice = {heuristic_choice}')
plt.title('Average Planning Time Total vs Action Space Size')
plt.xlabel('Action Space Size')
plt.ylabel('Average Planning Time Total')
plt.legend()
plt.grid(True)
plt.savefig(f'EXP_1_average_planning_time_plot_50_size=90_{time_str}.png')
plt.show()
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
from btgym.envs.RobotHow.exec_lib._base.RHAction import VHTAction
from btgym.algos.bt_autogen.tools import state_transition
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process, act_format_records
from btgym.algos.bt_autogen.Action import Action
from btgym.utils.tools import collect_action_nodes, save_data_txt
from btgym.utils.read_dataset import read_dataset
from btgym.algos.llm_client.llm_ask_tools import extract_llm_from_instr_goal
from tools import count_accuracy, identify_and_print_diffs, analyze_data_tabular,generate_custom_action_list
import matplotlib.pyplot as plt

env = btgym.make("VHT-Small")
# env = btgym.make("VHT-PutMilkInFridge")
cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
cur_cond_set |= {f'IsClose({arg})' for arg in VHTAction.CAN_OPEN}
cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHTAction.HAS_SWITCH}
cur_cond_set |= {f'IsUnplugged({arg})' for arg in VHTAction.HAS_PLUG}
big_actions = collect_action_nodes(env.behavior_lib)




# 导入数据
data_path = f"{ROOT_PATH}/../test/BT_EXP/data_small.txt"
data1 = read_dataset(data_path)
len_data = len(data1)
print(f"导入 {len_data} 条数据")
print(data1[0])
# analyze_data_tabular(data1,[40,1,1,1])



# 测试其中第1条
d = data1[0]
# goals = "IsClean_bananas & IsCut_bananas & IsIn_bananas_fridge"
goal_str = ' & '.join(d["Goals"])
# goal_str="IsOn_pear_kitchentable & IsOn_bananas_kitchentable"
goal_set = goal_transfer_str(goal_str)
print("goal_set:", goal_set)
d['Optimal Actions'] = act_str_process(d['Optimal Actions'], already_split=True)
# print("id:", " ", "goals:", d['Goals'])
print("Optimal Actions:", d['Optimal Actions'])

# Initialize lists to store individual results and averages
data_rows = []
action_space_sizes = list(range(15, 51, 5))


for action_space_size in action_space_sizes:
    print(f"\n============== action_space_sizes = {action_space_size} =============")
    custom_action_list = generate_custom_action_list(big_actions, action_space_size, d['Optimal Actions'])

    for heuristic_choice in [0, 1, -1]:
        if heuristic_choice == 0:
            print("----------priority = 0 的启发式 Running with Heuristic Choice = 0-----------")
        elif heuristic_choice == 1:
            print("--------priority = cost/10000 的启发式 Running with Heuristic Choice = 1---------")
        else:
            print("------------无启发式 Running with Heuristic Choice = -1--------")

        algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set, \
                              priority_act_ls=d['Optimal Actions'],  \
                              selected_algorithm="opt", mode="user-defined", action_list=custom_action_list,\
                              llm_reflect=False, time_limit=None,
                              heuristic_choice=heuristic_choice)
        # 定义变量 heuristic_choice：
        # 0 表示全是 0 的启发式
        # 1 表示 cost/10000 的启发式
        # -1 表示不使用启发式


        start_time = time.time()
        algo.process(goal_set)
        end_time = time.time()

        ptml_string, cost, expanded_num = algo.post_process()
        print("Expanded Conditions: ", expanded_num)
        planning_time_total = (end_time - start_time)
        print("planning_time_total:", planning_time_total)
        print("cost_total:", cost)

        time_limit_exceeded = algo.algo.time_limit_exceeded

        # Simulation and test
        goal = goal_set[0]
        state = cur_cond_set
        error, state, act_num, current_cost, record_act_ls = algo.execute_bt(goal, state, verbose=False)
        print(f"Executed {act_num - 1} action steps")
        print("current_cost:", current_cost)

        algo.algo.clear()

        # expanded_num = algo.post_process()[2]
        # planning_time_total = end_time - start_time
        #
        # goal = goal_set[0]
        # state = cur_cond_set
        # error, state, act_num, current_cost, record_act_ls = algo.execute_bt(goal, state, verbose=False)
        # time_limit_exceeded = algo.algo.time_limit_exceeded

        data_rows.append({
            'Goal': goal_str,
            'Optimal Actions' : d['Optimal Actions'],
            'Action_Space_Size': action_space_size,
            'Heuristic_Choice': heuristic_choice,
            'Expanded_Number': expanded_num,
            'Planning_Time_Total': planning_time_total,
            'Current_Cost': current_cost,
            'Error': error,
            'Time_Limit_Exceeded': time_limit_exceeded
        })

# Convert data to DataFrame
df = pd.DataFrame(data_rows)

# Save data to CSV
time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()).replace("-", "").replace(":", "")
df.to_csv(f'expanded_number_data_time={time_str}.csv', index=False)

# Plotting
plt.figure(figsize=(10, 6))
for heuristic_choice in [0, 1, -1]:
    plt.plot(df[df['Heuristic_Choice'] == heuristic_choice]['Action_Space_Size'],
             df[df['Heuristic_Choice'] == heuristic_choice]['Expanded_Number'],
             label=f'Heuristic Choice = {heuristic_choice}')
plt.title('Expanded Number vs Action Space Size')
plt.xlabel('Action Space Size')
plt.ylabel('Expanded Number')
plt.legend()
plt.grid(True)
plt.savefig(f'expanded_number_plot_time={time_str}.png')
plt.show()
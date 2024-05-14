import copy
import time

from btgym import BehaviorTree
import btgym
from btgym.utils import ROOT_PATH
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from btgym.envs.virtualhometext.exec_lib._base.VHTAction_small import VHTAction_small
from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction
from btgym.algos.bt_autogen.tools import state_transition
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process
import random
import numpy as np
import pandas as pd

seed = 0
random.seed(seed)
np.random.seed(seed)

from btgym.utils.tools import collect_action_nodes
from btgym.utils.read_dataset import read_dataset
from tools import count_accuracy, identify_and_print_diffs, analyze_data_tabular, generate_custom_action_list
import pickle

all_start_time = time.time()

# 导入数据
data_path = f"{ROOT_PATH}/../test/BT_EXP/EXP3_DATA_BT_100_ori_yz_revby_cys.txt"  # DATA_BT_100_ori_yz_revby_cys
data = read_dataset(data_path)
len_data = len(data)
print(f"导入 {len_data} 条数据")
print(data[0])
analyze_data_tabular(data, [47, 1, 1, 1])


# env = btgym.make("VHT-Small")
env = btgym.make("VHT-PutMilkInFridge")
cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
cur_cond_set |= {f'IsClose({arg})' for arg in VHTAction.CAN_OPEN}
cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHTAction.HAS_SWITCH}
cur_cond_set |= {f'IsUnplugged({arg})' for arg in VHTAction.HAS_PLUG}
big_actions = collect_action_nodes(env.behavior_lib)

# 把没有用到的动作 LeftPut 和 RightPut 都删掉
def filter_actions(data, big_actions):
    # Initialize an empty set to collect all unique action names from priority_act_ls across all data entries
    all_action_names = set()
    # Loop through the data to fill all_action_names with unique action names
    for ind, d in enumerate(data):
        priority_act_ls = act_str_process(d['Optimal Actions'], already_split=True)
        all_action_names.update(priority_act_ls)  # Add names to the set, ensuring uniqueness
    # Filter big_actions to find instances where the action name is in all_action_names
    current_big_actions = [action for action in big_actions if action.name in all_action_names]
    return current_big_actions
# Example usage:
# Assuming `big_actions` is already populated with Action instances and `data` is available
current_big_actions = filter_actions(data, big_actions)


# Prepare to collect results
results = []

for ind, d in enumerate(data):
    d = data[ind]
    combined_string_goal = ' & '.join(d['Goals'])
    goal_set = goal_transfer_str(combined_string_goal)
    priority_act_ls = act_str_process(d['Optimal Actions'], already_split=True)
    key_pred = d['Vital Action Predicates']
    key_obj = d['Vital Objects']
    # To print text in green
    print("\x1b[32m\ndata:", ind, "\x1b[0m")
    print("goal:", goal_set)
    print("act:", priority_act_ls)
    # print("key_pred:", key_pred)
    # print("key_obj:", key_obj)

    # mode = "user-defined" # 大空间
    # mode = "small-predicate-objs"
    for mode in ["user-defined","small-predicate-objs"]:
        for heuristic_choice in [-1,0,1]:

            algo = BTExpInterface(None, cur_cond_set=cur_cond_set,
                                  action_list=current_big_actions,\
                                  priority_act_ls=priority_act_ls, key_predicates=key_pred,
                                  key_objects=key_obj, \
                                  selected_algorithm="opt", mode=mode, \
                                  llm_reflect=False, time_limit=10,
                                  heuristic_choice=heuristic_choice)

            start_time = time.time()
            goal_set = goal_transfer_str(' & '.join(d["Goals"]))
            algo.process(goal_set)
            end_time = time.time()

            ptml_string, cost, expanded_num = algo.post_process()
            planning_time_total = (end_time - start_time)
            time_limit_exceeded = algo.algo.time_limit_exceeded

            # Simulation and test
            goal = goal_set[0]
            state = cur_cond_set
            error, state, act_num, current_cost, record_act_ls = algo.execute_bt(goal, state, verbose=False)
            # To print text in red
            # print("\x1b[31mdata:", ind, "\x1b[0m")
            print(f"\x1b[32mExecuted {act_num} action steps\x1b[0m", " \x1b[31mERROR\x1b[0m " if error else " ",
                  " \x1b[31mTIMEOUT\x1b[0m " if time_limit_exceeded else " ")
            print("current_cost", current_cost, "expanded_num:", expanded_num, "planning_time_total", planning_time_total)

            # Collecting data for export
            results.append({
                "Mode": mode,
                "Heuristic Choice": heuristic_choice,
                "Index": ind,
                "Goals": d['Goals'],
                "Optimal Actions": d['Optimal Actions'],
                "Expanded Num": expanded_num,
                "Planning Time Total": planning_time_total,
                "Current Cost": current_cost,
                "Error": error,
                "Time Limit Exceeded": time_limit_exceeded
            })


# Create DataFrame and save to Excel
results_df = pd.DataFrame(results)
results_df.to_csv(f"EXP_BT_3_big_small_3algo_table.csv", index=False)
print(f"Data has been saved to 'EXP_BT_3_big_small_3algo_table.csv")

# Filter for errors and time limit exceeded
errors = [result for result in results if result['Error']]
time_limits_exceeded = [result for result in results if result['Time Limit Exceeded']]

# Print counts and indices
print("---------------")
print(f"Total errors: {len(errors)}, Indices: {[result['Index'] for result in errors]}")
print(f"Total time limits exceeded: {len(time_limits_exceeded)}, Indices: {[result['Index'] for result in time_limits_exceeded]}")
print("---------------")

all_end_time = time.time()
total_time = (all_end_time - all_start_time)
print("Total Time:", total_time)

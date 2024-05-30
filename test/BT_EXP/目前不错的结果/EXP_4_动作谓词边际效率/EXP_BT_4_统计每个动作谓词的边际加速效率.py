import copy
import time

from btgym import BehaviorTree
import btgym
from btgym.utils import ROOT_PATH
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from btgym.envs.RobotHow.exec_lib._base.VHTAction_small import VHTAction_small
from btgym.envs.RobotHow.exec_lib._base.RHAction import VHTAction
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
data_path = f"{ROOT_PATH}/../test/BT_EXP/DATA_BT_100_ori_yz_revby_cys.txt"  # DATA_BT_100_ori_yz_revby_cys
# data_path = f"{ROOT_PATH}/../test/BT_EXP/EXP3_DATA_BT_100_ori_yz_revby_cys_003.txt"  # DATA_BT_100_ori_yz_revby_cys
data = read_dataset(data_path)
len_data = len(data)
print(f"导入 {len_data} 条数据")
print(data[0])
analyze_data_tabular(data, [47, 1, 1, 1])

env = btgym.make("VHT-PutMilkInFridge")
cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
cur_cond_set |= {f'IsClose({arg})' for arg in VHTAction.CAN_OPEN}
cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHTAction.HAS_SWITCH}
cur_cond_set |= {f'IsUnplugged({arg})' for arg in VHTAction.HAS_PLUG}
# big_actions = collect_action_nodes(env.behavior_lib)

# Helper functions you might need
def execute_algorithm(goal_set, predicates, objects, priority_actions):
    """
    Simulate the algorithm execution with given settings and return the metrics.
    """

    algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                          priority_act_ls=priority_actions, key_predicates=predicates,
                          key_objects=objects, selected_algorithm="opt",
                          mode="small-predicate-objs", llm_reflect=False, time_limit=5,
                          heuristic_choice=0)

    start_time = time.time()
    algo.process(goal_set)
    end_time = time.time()

    ptml_string, cost, expanded_num = algo.post_process()
    planning_time_total = end_time - start_time
    goal = goal_set[0]
    error, state, act_num, current_cost, record_act_ls = algo.execute_bt(goal, cur_cond_set, verbose=False)

    time_limit_exceeded = algo.algo.time_limit_exceeded
    return expanded_num, planning_time_total, current_cost, error, time_limit_exceeded, act_num


# Initialize data structures
results = []
predicate_effects = {}


for ind, d in enumerate(data):

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


    # Record baseline metrics with all predicates
    # Execute with all predicates
    expanded_num, planning_time_total, current_cost, error, time_limit_exceeded, act_num = \
        execute_algorithm(goal_set,key_pred,key_obj,priority_act_ls)

    # Print baseline metrics
    print(f"Before: \x1b[32mExecuted {act_num} action steps\x1b[0m", "\x1b[31mERROR\x1b[0m" if error else "",
          "\x1b[31mTIMEOUT\x1b[0m" if time_limit_exceeded else "")
    print("current_cost:", current_cost, "expanded_num:", expanded_num, "planning_time_total:", planning_time_total)

    for pred in key_pred:
        # Modify action list by removing current predicate
        modified_priority_actions = [act for act in priority_act_ls if pred not in act]

        # Execute without the current predicate
        modified_expanded_num, modified_planning_time_total, modified_current_cost, modified_error, modified_time_limit_exceeded, modified_act_num = execute_algorithm(
            goal_set, key_pred, key_obj, modified_priority_actions)

        # Print metrics after removal
        print(f"After removing {pred}: \x1b[32mExecuted {modified_act_num} action steps\x1b[0m",
              "\x1b[31mERROR\x1b[0m" if modified_error else "",
              "\x1b[31mTIMEOUT\x1b[0m" if modified_time_limit_exceeded else "")
        print("current_cost:", modified_current_cost, "expanded_num:", modified_expanded_num, "planning_time_total:",
              modified_planning_time_total)

        # Calculate and store differences
        metrics_diff = {
            'expanded_num_diff': modified_expanded_num - expanded_num,
            'time_diff': modified_planning_time_total - planning_time_total,
            'cost_diff': modified_current_cost - current_cost,
            'error_diff': modified_error- error,
            'timeout_diff': modified_time_limit_exceeded - time_limit_exceeded
        }

        # Store results
        results.append({
            'predicate': pred,
            'metrics_diff': metrics_diff
        })

        # Update global stats for predicate
        if pred not in predicate_effects:
            predicate_effects[pred] = [metrics_diff]
        else:
            predicate_effects[pred].append(metrics_diff)

# First, aggregate the effects for each predicate
for pred, effects in predicate_effects.items():
    avg_effects = pd.DataFrame(effects).mean().to_dict()
    predicate_effects[pred] = avg_effects

# Convert the dictionary of effects to a DataFrame
effects_df = pd.DataFrame.from_dict(predicate_effects, orient='index')
# Sort the DataFrame by 'expanded_num_diff' from high to low
sorted_effects_df = effects_df.sort_values(by='expanded_num_diff', ascending=False)
# Print sorted DataFrame
print(sorted_effects_df)
# Save the sorted DataFrame to CSV if needed
sorted_effects_df.to_csv("EXP_BT_4_predicate_effects.csv")

# If visualization is needed:
import matplotlib.pyplot as plt
import seaborn as sns
# Visualize the sorted effects using a bar plot
plt.figure(figsize=(10, 8))
sns.barplot(x='expanded_num_diff', y=sorted_effects_df.index, data=sorted_effects_df)
plt.title('Impact of Predicates on Expanded Number Difference')
plt.xlabel('Average Difference in Expanded Nodes')
plt.ylabel('Predicates')
plt.show()

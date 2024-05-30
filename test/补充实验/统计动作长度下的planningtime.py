import re
import time
import random
import numpy as np
import pandas as pd
import pickle
from tools import execute_algorithm, load_dataset, setup_default_env
from sympy import symbols, Not, Or, And, to_dnf, simplify_logic
import btgym
from btgym import BehaviorTree, ExecBehaviorLibrary
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
from tools import count_accuracy, identify_and_print_diffs, analyze_data_tabular
import matplotlib.pyplot as plt

# Set random seed for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)

# Initialize environment
env = btgym.make("VHT-PutMilkInFridge")
cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
cur_cond_set |= {f'IsClose({arg})' for arg in VHTAction.CAN_OPEN}
cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHTAction.HAS_SWITCH}
cur_cond_set |= {f'IsUnplugged({arg})' for arg in VHTAction.HAS_PLUG}
big_actions = collect_action_nodes(env.behavior_lib)

# Import data
data_path = f"new_data.txt"
data1 = read_dataset(data_path)
len_data = len(data1)
print(f"Imported {len_data} data entries")
print(data1[0])
analyze_data_tabular(data1, [1, 1, 1, 1])

def algo_h(chosen_goal, priority_act_ls, key_predicates, key_objects, heuristic_choice):
    algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                          priority_act_ls=priority_act_ls, key_predicates=key_predicates,
                          key_objects=key_objects,
                          selected_algorithm="opt", mode="small-predicate-objs",
                          llm_reflect=False, time_limit=7,
                          heuristic_choice=heuristic_choice)
    goal_set = goal_transfer_str(' & '.join(chosen_goal))
    expanded_num, planning_time_total, cost, error, act_num, current_cost, record_act_ls = \
        execute_algorithm(algo, goal_set, cur_cond_set)
    time_limit_exceeded = algo.algo.time_limit_exceeded
    success = not error and not time_limit_exceeded
    act_space = len(algo.actions)

    if time_limit_exceeded:
        RED = "\033[31m"
        RESET = "\033[0m"
        print(f"{RED}- H: {heuristic_choice} Goal:{chosen_goal}  Time Out  -{RESET}")
    elif error:
        RED = "\033[31m"
        RESET = "\033[0m"
        print(f"{RED}- H: {heuristic_choice} Goal:{chosen_goal}  Error  -{RESET}")

    return expanded_num, planning_time_total, current_cost, success

def save_data_to_file(id, data, goal_str, formatted_act, formatted_predicates, formatted_objects, current_cost, output_path, need_cost=False):
    entry_str = f"{id}\n"
    entry_str += "Environment:1\n"
    entry_str += f"Instruction: {data['Instruction']}\n"
    entry_str += f"Goals: {goal_str}\n"
    entry_str += f"Optimal Actions: {formatted_act}\n"
    entry_str += f"Vital Action Predicates: {formatted_predicates}\n"
    entry_str += f"Vital Objects: {formatted_objects}\n"
    if need_cost:
        entry_str += f"cost: {str(current_cost)}\n"
    with open(output_path, "a") as f:
        f.write(entry_str + "\n")
    print("Written to file:", entry_str)

# Dictionary to store execution times and other metrics for different action lengths
execution_metrics = {}
output_path = "processed_data.txt"

# Iterate through each data entry and compute execution metrics
for i, d in enumerate(data1):
    goal_str = ' & '.join(d["Goals"])
    goal_set = goal_transfer_str(goal_str)
    print(f"i: {i}, goal_set: {goal_set}")
    d['Optimal Actions'] = [v.strip() for v in d['Optimal Actions'][0].split("&")]
    priority_act_ls = act_str_process(d['Optimal Actions'], already_split=True)
    print(f"Optimal Actions: {d['Optimal Actions']}")

    act_len = len(priority_act_ls)

    # Define placeholders for priority_act_ls, key_predicates, key_objects

    key_predicates = d["Vital Action Predicates"]
    key_objects = d["Vital Objects"]

    h0_metrics = algo_h(d["Goals"], priority_act_ls, key_predicates, key_objects, 0)
    if not h0_metrics[3]:
        continue
    h1_metrics = algo_h(d["Goals"], priority_act_ls, key_predicates, key_objects, 1)
    if not h1_metrics[3] or not h0_metrics[3]:
        continue  # Skip if there is a timeout or error

    if act_len not in execution_metrics:
        execution_metrics[act_len] = {'h1': {'expanded_num': [], 'planning_time_total': [], 'current_cost': []},
                                      'h0': {'expanded_num': [], 'planning_time_total': [], 'current_cost': []}}
    execution_metrics[act_len]['h1']['expanded_num'].append(h1_metrics[0])
    execution_metrics[act_len]['h1']['planning_time_total'].append(h1_metrics[1])
    execution_metrics[act_len]['h1']['current_cost'].append(h1_metrics[2])
    execution_metrics[act_len]['h0']['expanded_num'].append(h0_metrics[0])
    execution_metrics[act_len]['h0']['planning_time_total'].append(h0_metrics[1])
    execution_metrics[act_len]['h0']['current_cost'].append(h0_metrics[2])

    # Save non-timeout data to file
    formatted_act = ' & '.join(d['Optimal Actions'])
    formatted_predicates = ','.join(d["Vital Action Predicates"])
    formatted_objects = ','.join(d["Vital Objects"])
    save_data_to_file(i, d, goal_str, formatted_act, formatted_predicates, formatted_objects, h1_metrics[2], output_path)

# Sort action lengths
action_lengths = sorted(execution_metrics.keys())

# Get average metrics sorted by action lengths
h1_avg_expanded_num = [np.mean(execution_metrics[length]['h1']['expanded_num']) for length in action_lengths]
h0_avg_expanded_num = [np.mean(execution_metrics[length]['h0']['expanded_num']) for length in action_lengths]
h1_avg_planning_time = [np.mean(execution_metrics[length]['h1']['planning_time_total']) for length in action_lengths]
h0_avg_planning_time = [np.mean(execution_metrics[length]['h0']['planning_time_total']) for length in action_lengths]
h1_avg_cost = [np.mean(execution_metrics[length]['h1']['current_cost']) for length in action_lengths]
h0_avg_cost = [np.mean(execution_metrics[length]['h0']['current_cost']) for length in action_lengths]
num_data_points = [len(execution_metrics[length]['h1']['expanded_num']) for length in action_lengths]

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'Action Length': action_lengths,
    'H1 Avg Expanded Num': h1_avg_expanded_num,
    'H0 Avg Expanded Num': h0_avg_expanded_num,
    'H1 Avg Planning Time': h1_avg_planning_time,
    'H0 Avg Planning Time': h0_avg_planning_time,
    'H1 Avg Cost': h1_avg_cost,
    'H0 Avg Cost': h0_avg_cost,
    'Num Data Points': num_data_points
})

# Save results to a CSV file
results_df.to_csv('algorithm_comparison_results.csv', index=False)

# Plot average expanded_num vs action length
plt.figure(figsize=(10, 6))
plt.plot(action_lengths, h1_avg_expanded_num, label='H1 Algorithm - Expanded Num', marker='o')
plt.plot(action_lengths, h0_avg_expanded_num, label='H0 Algorithm - Expanded Num', marker='o')
plt.xlabel('Action Length')
plt.ylabel('Average Expanded Num')
plt.title('Average Expanded Num vs Action Length for H1 and H0 Algorithms')
plt.legend()
plt.grid(True)
plt.show()

# Plot average planning_time_total vs action length
plt.figure(figsize=(10, 6))
plt.plot(action_lengths, h1_avg_planning_time, label='H1 Algorithm - Planning Time', marker='o')
plt.plot(action_lengths, h0_avg_planning_time, label='H0 Algorithm - Planning Time', marker='o')
plt.xlabel('Action Length')
plt.ylabel('Average Planning Time (seconds)')
plt.title('Average Planning Time vs Action Length for H1 and H0 Algorithms')
plt.legend()
plt.grid(True)
plt.show()

# Plot average current_cost vs action length
plt.figure(figsize=(10, 6))
plt.plot(action_lengths, h1_avg_cost, label='H1 Algorithm - Cost', marker='o')
plt.plot(action_lengths, h0_avg_cost, label='H0 Algorithm - Cost', marker='o')
plt.xlabel('Action Length')
plt.ylabel('Average Cost')
plt.title('Average Cost vs Action Length for H1 and H0 Algorithms')
plt.legend()
plt.grid(True)
plt.show()

# Plot the number of data points for each action length
plt.figure(figsize=(10, 6))
plt.bar(action_lengths, num_data_points, color='blue')
plt.xlabel('Action Length')
plt.ylabel('Number of Data Points')
plt.title('Number of Data Points vs Action Length')
plt.grid(True)
plt.show()

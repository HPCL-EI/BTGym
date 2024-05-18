import time
import re
import os
import random
import numpy as np
import pandas as pd
from itertools import chain
from btgym import BehaviorTree, ExecBehaviorLibrary
import btgym
from btgym.utils import ROOT_PATH
from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction
from sympy import symbols, Not, Or, And, to_dnf, simplify_logic
from btgym.utils.read_dataset import read_dataset
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process, act_format_records
from btgym.utils.tools import collect_action_nodes

seed = 0
random.seed(seed)
np.random.seed(seed)

# 使用正则表达式从谓词中抽取宾语
def extract_objects(actions):
    pattern = re.compile(r'\w+\(([^)]+)\)')
    objects = []
    for action in actions:
        match = pattern.search(action)
        if match:
            objects.append(match.group(1))
    return objects

# 直接读入 env=1 的数据
# file_name = "test_data_40_0518_no_processed"
# file_name = "test_data_40_0518_2_no_processed"

# data_path = f"{ROOT_PATH}/../test/VD_EXP/{file_name}.txt"
# data_path = f"{ROOT_PATH}/../test/dataset/{file_name}.txt"
# output_path = f"{ROOT_PATH}/../test/VD_EXP/{file_name}_processed_data.txt"
# output_csv_path = f"{ROOT_PATH}/../test/LLM_EXP/{file_name}_processed_h=0.csv"

file_name="VS"
data_path = f"{ROOT_PATH}/../test/SCENES_EXP/{file_name}.txt"
output_path = f"{ROOT_PATH}/../test/SCENES_EXP/{file_name}_processed_data.txt"
output_csv_path = f"{ROOT_PATH}/../test/SCENES_EXP/{file_name}_processed_h=1.csv"
need_cost = True
data1 = read_dataset(data_path)
len_data = len(data1)
print(f"导入 {len_data} 条数据")
print(data1[0])

# 初始化环境
# env = btgym.make("VHT-PutMilkInFridge")
# cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
# cur_cond_set |= {f'IsClose({arg})' for arg in VHTAction.CAN_OPEN}
# cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHTAction.HAS_SWITCH}
# cur_cond_set |= {f'IsUnplugged({arg})' for arg in VHTAction.HAS_PLUG}
# big_actions = collect_action_nodes(env.behavior_lib)


from btgym.envs.virtualhome.exec_lib._base.VHAction import VHAction
env = btgym.make("VH-PutMilkInFridge")
cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
cur_cond_set |= {f'IsClose({arg})' for arg in VHAction.CanOpenPlaces}
cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHAction.HasSwitchObjects}
big_actions = collect_action_nodes(env.behavior_lib)


# 过滤动作
def filter_actions(data, big_actions):
    all_action_names = set()
    for d in data:
        print(d)
        priority_act_ls = act_str_process(d['Optimal Actions'], already_split=True)
        all_action_names.update(priority_act_ls)
    current_big_actions = [action for action in big_actions if action.name in all_action_names]
    current_big_actions_name = [action.name for action in big_actions if action.name in all_action_names]
    return current_big_actions, current_big_actions_name

current_big_actions, current_big_actions_name = filter_actions(data1, big_actions)

print("-----------------准备写入文件-----------------")
if os.path.exists(output_path):
    os.remove(output_path)

results = []
errors = []
time_limits_exceeded = []

def write_to_file(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a') as file:
        file.write(data + '\n')

for id, d in enumerate(data1):
    print("\x1b[32m\ndata:", id, "\x1b[0m", d["Instruction"])

    goal_str = ' & '.join(d["Goals"])
    act_str = ', '.join(d["Optimal Actions"])

    goal_set = goal_transfer_str(goal_str)
    print("goal_set:", goal_set)
    priority_act_ls = act_str_process(act_str)
    print("priority_act_ls:", priority_act_ls)

    key_predicates = extract_objects(priority_act_ls)

    priority_obj_ls = []
    objects = set()
    pattern = re.compile(r'\((.*?)\)')
    for expr in chain(goal_set[0], priority_act_ls):
        match = pattern.search(expr)
        if match:
            objects.update(match.group(1).split(','))
    priority_obj_ls += list(objects)

    # algo = BTExpInterface(None, cur_cond_set=cur_cond_set,
    #                       action_list=current_big_actions,
    #                       priority_act_ls=priority_act_ls, key_predicates=key_predicates,
    #                       key_objects=priority_obj_ls,
    #                       selected_algorithm="opt", mode="small-predicate-objs",
    #                       llm_reflect=False, time_limit=300,
    #                       heuristic_choice=1)

    algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                          priority_act_ls=priority_act_ls, key_predicates=key_predicates,
                          key_objects=priority_obj_ls,
                          selected_algorithm="opt", mode="small-predicate-objs",
                          llm_reflect=False, time_limit=None,
                          heuristic_choice=1)

    start_time = time.time()
    algo.process(goal_set)
    end_time = time.time()
    planning_time_total = end_time - start_time

    time_limit_exceeded = algo.algo.time_limit_exceeded

    ptml_string, cost, expanded_num = algo.post_process()
    error, state, act_num, current_cost, record_act_ls = algo.execute_bt(goal_set[0], cur_cond_set, verbose=False)

    print(f"\x1b[32mExecuted {act_num} action steps\x1b[0m",
          "\x1b[31mERROR\x1b[0m" if error else "",
          "\x1b[31mTIMEOUT\x1b[0m" if time_limit_exceeded else "")
    print("current_cost:", current_cost, "expanded_num:", expanded_num, "planning_time_total:", planning_time_total)

    correct_act, predicate, objects = act_format_records(record_act_ls)

    def extract_and_format(items):
        return ', '.join(items)
    formatted_act = extract_and_format(correct_act)
    formatted_predicates = extract_and_format(predicate)
    formatted_objects = extract_and_format(objects)

    print("Goals:", goal_str)
    print("Optimal Actions:", formatted_act)

    priority_act = set(act_str.replace(" ", "").split(","))
    print("增加了：", set(correct_act) - priority_act)
    print("减少了：", priority_act - set(correct_act))

    entry_str = f"{id+1}\n"
    entry_str += f"Environment:1\n"
    entry_str += f"Instruction: {d['Instruction']}\n"
    entry_str += f"Goals: {goal_str}\n"
    entry_str += f"Optimal Actions: {formatted_act}\n"
    entry_str += f"Vital Action Predicates: {formatted_predicates}\n"
    entry_str += f"Vital Objects: {formatted_objects}\n"
    if need_cost:
        entry_str += f"cost: {str(current_cost)}\n"

    write_to_file(entry_str, output_path)
    print("Written to file:", entry_str)

    results.append({
        'id': id + 1,
        'Environment': 1,
        'Instruction': d['Instruction'],
        'Goals': goal_str,
        'Optimal Actions': formatted_act,
        'Vital Action Predicates': formatted_predicates,
        'Vital Objects': formatted_objects,
        'expanded_num': expanded_num,
        'planning_time_total': planning_time_total,
        'current_cost': current_cost,
        'error': error,
        'act_num': act_num
    })

    if error or time_limit_exceeded:
        if error:
            errors.append({
                'id': id + 1,
                'Goals': goal_str
            })
        if time_limit_exceeded:
            time_limits_exceeded.append({
                'id': id + 1,
                'Goals': goal_str
            })

df = pd.DataFrame(results)
df.to_csv(output_csv_path, index=False)
print(f"Results have been saved to {output_csv_path}")

# Print counts and indices
print("---------------")
print(f"Total errors: {len(errors)}, Indices: {[error['id'] for error in errors]}, Goals: {[error['Goals'] for error in errors]}")
print(f"Total time limits exceeded: {len(time_limits_exceeded)}, Indices: {[tl['id'] for tl in time_limits_exceeded]}, Goals: {[tl['Goals'] for tl in time_limits_exceeded]}")
print("---------------")

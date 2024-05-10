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
from btgym.envs.virtualhometext.exec_lib._base.VHTAction_small import VHTAction_small
from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction
from btgym.algos.bt_autogen.tools import state_transition
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process, act_format_records
from btgym.algos.bt_autogen.Action import Action
from btgym.utils.tools import collect_action_nodes, save_data_txt
from btgym.utils.read_dataset import read_dataset
from btgym.algos.llm_client.llm_ask_tools import extract_llm_from_instr_goal
from tools import count_accuracy, identify_and_print_diffs,analyze_data_tabular

# # 读入数据集
# data_path = f"{ROOT_PATH}/../test/dataset/data0506/dataset0506_revby_cys.txt"
# data = read_dataset(data_path)
# len_data = len(data)
# print(f"导入 {len_data} 条数据")
# print(data[0])
#
# # 挑选出 env=1 的数据进行测试，总共 40条
# data1 = [d for d in data if d['Environment'] == 1]
# len_data = len(data1)
# print(f"环境为1的数据总共有 {len_data} 条")
# output_path = f"{ROOT_PATH}/../test/dataset/data1_env1_40.txt"
# save_data_txt(output_path, data1)

# 直接读入 env=1 的数据
data_path = f"{ROOT_PATH}/../test/dataset/data1_env1_40_test_reflect.txt"
data1 = read_dataset(data_path)
len_data = len(data1)
print(f"导入 {len_data} 条数据")
print(data1[0])

# 打印数据信息
analyze_data_tabular(data1,[10,10,10,10])


# 导入 prompt，根据 instruction 和 goal 得到其它三项
# LLM
llm = LLMGPT3()
# llm=LLMGPT4()
default_prompt_file = f"{ROOT_PATH}\\algos\\llm_client\\prompt_VHT_just_goal.txt"

results = []
start_time = time.time()
# 解析这三项，得到正确率错误率
for id, d in enumerate(data1[:0]):
    print("id:", id, "  ", d['Instruction'])
    print("id:",id ," ","goals:",d['Goals'])

    # instuction = "Wash the bananas, cut the bananas and put it in the fridge"
    # goals = "IsClean_bananas & IsCut_bananas & IsIn_bananas_fridge"
    instruction = d['Instruction']
    goals = d['Goals']
    d['Optimal Actions'] = act_str_process(d['Optimal Actions'])

    # 大模型推荐的结果
    priority_act_ls, llm_key_pred, llm_key_obj, messages = \
        extract_llm_from_instr_goal(llm, default_prompt_file, instruction, goals, verbose=False)
    # print("------------------------")
    print("Act:", priority_act_ls)
    print("Key_Predicate", llm_key_pred)
    print("Vital Objects:", llm_key_obj)

    # key_predicates 和 key_objects 要将推荐的 priority_act_ls 补充进来
    _, pred, obj = act_format_records(priority_act_ls)
    key_predicates = list(set(llm_key_pred + pred))
    key_objects = list(set(llm_key_obj + obj))

    # 统计动作的准确率和错误率
    act_correct, act_incorrect, act_accuracy, act_error_rate = count_accuracy(d['Optimal Actions'], priority_act_ls)

    # 统计关键谓词的准确率和错误率
    predicate_correct, predicate_incorrect, predicate_accuracy, predicate_error_rate = count_accuracy(
        d['Vital Action Predicates'], key_predicates)

    # 统计关键对象的准确率和错误率
    object_correct, object_incorrect, object_accuracy, object_error_rate = count_accuracy(d['Vital Objects'],
                                                                                          key_objects)

    # 打印统计结果
    print(f"Act:  {act_accuracy:.3f}%")
    print(f"Pred: {predicate_accuracy:.3f}%")
    print(f"Objs: {object_accuracy:.3f}%")
    # Identify missed and incorrectly predicted actions/predicates/objects
    diff = identify_and_print_diffs(d, priority_act_ls, key_predicates, key_objects)

    # 先判断三个的结果对不对
    # if not diff:
    #     break

    #  =============== 如果不对，跑一下 BT==================
    env = btgym.make("VHT-PutMilkInFridge")
    cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
    cur_cond_set |= {f'IsClose({arg})' for arg in VHTAction.CAN_OPEN}
    cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHTAction.HAS_SWITCH}
    cur_cond_set |= {f'IsUnplugged({arg})' for arg in VHTAction.HAS_PLUG}
    algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set, \
                          priority_act_ls=priority_act_ls, key_predicates=key_predicates, key_objects=key_objects, \
                          selected_algorithm="opt", mode='small-predicate-objs', \
                          llm_reflect=False, use_priority_act=True, time_limit=10)

    start_time = time.time()
    goal_str = ' & '.join(d["Goals"])
    goal_set = goal_transfer_str(goal_str)
    algo.process(goal_set)
    end_time = time.time()

    ptml_string, cost, expanded_num = algo.post_process()
    print("Expanded Conditions: ", expanded_num)
    planning_time_total = (end_time - start_time)
    print("planning_time_total:", planning_time_total)
    print("cost_total:", cost)

    time_limit_exceeded = algo.algo.time_limit_exceeded
    if time_limit_exceeded:
        RED = "\033[31m"
        RESET = "\033[0m"
        print(f"{RED}---Error: 设定不超过 30 s, 超时停止！----{RESET}")

    # Simulation and test
    print("\n================ ")
    goal = goal_set[0]
    state = cur_cond_set
    error, state, act_num, current_cost, record_act_ls = algo.execute_bt(goal, state, verbose=True)
    if error:
        RED = "\033[31m"
        RESET = "\033[0m"
        print(f"{RED}---Error----{RESET}")
    print(f"Executed {act_num - 1} action steps")
    print("current_cost:", current_cost)
    print("================ ")
    #  =============== 如果不对，跑一下 BT==================

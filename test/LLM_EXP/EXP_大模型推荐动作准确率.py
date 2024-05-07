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
from tools import count_accuracy

# # 读入数据集
# data_path = f"{ROOT_PATH}/../test/dataset/dataset0506/dataset0506_revby_cys.txt"
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
data_path = f"{ROOT_PATH}/../test/dataset/data1_env1_40.txt"
data1 = read_dataset(data_path)
len_data = len(data1)
print(f"导入 {len_data} 条数据")
print(data1[0])

# 导入 prompt，根据 instruction 和 goal 得到其它三项
# LLM
# llm = LLMGPT3()
llm=LLMGPT4()
# default_prompt_file = f"{ROOT_PATH}\\algos\\llm_client\\prompt_VHT_goal.txt"
default_prompt_file = f"{ROOT_PATH}\\algos\\llm_client\\prompt_VHT_just_goal.txt"

results = []
start_time = time.time()
# 解析这三项，得到正确率错误率
for id, d in enumerate(data1):
    # print("id:", id)
    print("id:", id, "  ", d['Instruction'])
    # instuction = "Wash the bananas, cut the bananas and put it in the fridge"
    # goals = "IsClean_bananas & IsCut_bananas & IsIn_bananas_fridge"

    instruction = d['Instruction']
    goals = d['Goals']
    d['Actions'] = act_str_process(d['Actions'], already_split=True)

    # 大模型推荐的结果
    priority_act_ls, llm_key_pred, llm_key_obj, messages = \
        extract_llm_from_instr_goal(llm, default_prompt_file, instruction, goals, verbose=False)
    # print("------------------------")
    print("Act:", priority_act_ls)
    print("Key_Predicate", llm_key_pred)
    print("Key_Objects:", llm_key_obj)

    # key_predicates 和 key_objects 要将推荐的 priority_act_ls 补充进来
    _, pred, obj = act_format_records(priority_act_ls)
    key_predicates = list(set(llm_key_pred + pred))
    key_objects = list(set(llm_key_obj + obj))

    # 统计动作的准确率和错误率
    act_correct, act_incorrect, act_accuracy, act_error_rate = count_accuracy(d['Actions'], priority_act_ls)

    # 统计关键谓词的准确率和错误率
    predicate_correct, predicate_incorrect, predicate_accuracy, predicate_error_rate = count_accuracy(
        d['Key_Predicates'], key_predicates)

    # 统计关键对象的准确率和错误率
    object_correct, object_incorrect, object_accuracy, object_error_rate = count_accuracy(d['Key_Objects'], key_objects)

    # 打印统计结果
    print("Actions:")
    print(f"  Accuracy: {act_accuracy:.3f}%")
    print("Key Predicates:")
    print(f"  Accuracy: {predicate_accuracy:.3f}%")
    print("Key Objects:")
    print(f"  Accuracy: {object_accuracy:.3f}%")

    # Identify missed and incorrectly predicted actions/predicates/objects
    missing_act = set(d['Actions']) - set(priority_act_ls)
    incorrect_act = set(priority_act_ls) - set(d['Actions'])

    missing_predicate = set(d['Key_Predicates']) - set(key_predicates)
    incorrect_predicate = set(key_predicates) - set(d['Key_Predicates'])

    missing_obj = set(d['Key_Objects']) - set(key_objects)
    incorrect_obj = set(key_objects) - set(d['Key_Objects'])

    # Prepare the results
    results.append({
        'ID': id,  # Add 1 to make it 1-indexed instead of 0-indexed
        'Instruction': instruction, 'Goals': goals, 'Acts (Exp)': d['Actions'],
        'Preds (Exp)': d['Key_Predicates'], 'Objs (Exp)': d['Key_Objects'],
        'Acts (LLM)': priority_act_ls, 'Preds (LLM)': key_predicates, 'Objs (LLM)': key_objects,
        'Preds (Just LLM)': llm_key_pred, 'Objs (Just LLM)': llm_key_obj,
        'Increase_Preds': ', '.join(set(key_predicates) - set(llm_key_pred)),
        'Increase_Objs':  ', '.join(set(key_objects) - set(llm_key_obj)),

        'Act_Count (Exp)': len(d['Actions']), 'Act_Count (LLM)': len(priority_act_ls), 'Correct_Act': act_correct,
        'Act_Acc': act_accuracy, 'Missed_Act': ', '.join(missing_act), 'Incorrect_Act': ', '.join(incorrect_act),

        'Pred_Count (Exp)': len(d['Key_Predicates']), 'Pred_Count (LLM)': len(key_predicates),
        'Correct_Pred': predicate_correct,
        'Pred_Acc': predicate_accuracy, 'Missed_Pred': ', '.join(missing_predicate),
        'Incorrect_Pred': ', '.join(incorrect_predicate),

        'Obj_Count (Exp)': len(d['Key_Objects']), 'Obj_Count (LLM)': len(key_objects), 'Correct_Obj': object_correct,
        'Obj_Acc': object_accuracy, 'Missed_Obj': ', '.join(missing_obj), 'Incorrect_Obj': ', '.join(incorrect_obj),

    })

# Convert results list to a DataFrame
results_df = pd.DataFrame(results)
time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()).replace("-", "").replace(":", "")
results_df.to_csv(f"gpt4_40_time={time_str}.csv", index=False)
print(f"Results have been saved to gpt4_40_time={time_str}.csv")
end_time = time.time()
time_total = (end_time - start_time)
print("Total time:",time_total)
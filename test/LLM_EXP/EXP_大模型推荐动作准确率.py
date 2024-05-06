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
from btgym.algos.bt_autogen.main_interface import BTExpInterface, collect_action_nodes
from btgym.envs.virtualhometext.exec_lib._base.VHTAction_small import VHTAction_small
from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction
from btgym.algos.bt_autogen.tools import state_transition
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process
from btgym.algos.bt_autogen.Action import Action
from btgym.utils.tools import collect_action_nodes
from btgym.utils.read_dataset import read_dataset
from btgym.algos.llm_client.llm_ask_tools import extract_llm_from_instr_goal
from tools import count_accuracy

# 读入数据集
data_path = f"{ROOT_PATH}/../test/dataset/dataset0506/dataset0506.txt"
data = read_dataset(data_path)
len_data = len(data)
print(f"导入 {len_data} 条数据")
print(data[0])

# 挑选出 env=1 的数据进行测试，总共 40条
data1 = [d for d in data if d['Environment'] == 1]
len_data = len(data1)
print(f"环境为1的数据总共有 {len_data} 条")


# 导入 prompt，根据 instruction 和 goal 得到其它三项
# LLM
llm = LLMGPT3()
default_prompt_file = f"{ROOT_PATH}\\algos\\llm_client\\prompt_VHT_goal.txt"

# 输出到 excel 中
import pandas as pd

# 创建一个空的DataFrame用于存储结果
results_df = pd.DataFrame(columns=['Instruction', 'Goals', 'Expected Actions', 'Expected Key Predicates', 'Expected Key Objects',
                                   'LLM Output Actions', 'LLM Output Key Predicates', 'LLM Output Key Objects',
                                   'Actions Correct', 'Actions Incorrect', 'Actions Accuracy', 'Actions Error Rate',
                                   'Key Predicates Correct', 'Key Predicates Incorrect', 'Key Predicates Accuracy', 'Key Predicates Error Rate',
                                   'Key Objects Correct', 'Key Objects Incorrect', 'Key Objects Accuracy', 'Key Objects Error Rate'])


# 解析这三项，得到正确率错误率
for id,d in enumerate(data1):
    print("id:",id)
    # instuction = "Wash the bananas, cut the bananas and put it in the fridge"
    # goals = "IsClean_bananas & IsCut_bananas & IsIn_bananas_fridge"

    instruction = d['Instruction']
    goals = d['Goals']

    priority_act_ls, key_predicates, key_objects,messages = \
        extract_llm_from_instr_goal(llm,default_prompt_file,instruction,goals,verbose=False)
    # print("------------------------")
    # print("Act:", priority_act_ls)
    # print("Key_Predicate", key_predicates)
    # print("Key_Objects:", key_objects)


    # 统计动作的准确率和错误率
    act_correct, act_incorrect, act_accuracy, act_error_rate = count_accuracy(d['Actions'], priority_act_ls)

    # 统计关键谓词的准确率和错误率
    predicate_correct, predicate_incorrect, predicate_accuracy, predicate_error_rate = count_accuracy(d['Key_Predicate'], key_predicates)

    # 统计关键对象的准确率和错误率
    object_correct, object_incorrect, object_accuracy, object_error_rate = count_accuracy(d['Key_Object'], key_objects)

    # 打印统计结果
    print("Actions:")
    # print(f"  Correct: {act_correct}")
    # print(f"  Incorrect: {act_incorrect}")
    print(f"  Accuracy: {act_accuracy}%")
    # print(f"  Error rate: {act_error_rate}%\n")

    print("Key Predicates:")
    # print(f"  Correct: {predicate_correct}")
    # print(f"  Incorrect: {predicate_incorrect}")
    print(f"  Accuracy: {predicate_accuracy}%")
    # print(f"  Error rate: {predicate_error_rate}%\n")

    print("Key Objects:")
    # print(f"  Correct: {object_correct}")
    # print(f"  Incorrect: {object_incorrect}")
    print(f"  Accuracy: {object_accuracy}%")
    # print(f"  Error rate: {object_error_rate}%")

   # 将结果存入DataFrame
    results_df = results_df.append({'Instruction': instruction,
                                    'Goals': goals,
                                    'Expected Actions': d['Actions'],
                                    'Expected Key Predicates': d['Key_Predicate'],
                                    'Expected Key Objects': d['Key_Object'],
                                    'LLM Output Actions': priority_act_ls,
                                    'LLM Output Key Predicates': key_predicates,
                                    'LLM Output Key Objects': key_objects,
                                    'Actions Correct': act_correct,
                                    'Actions Incorrect': act_incorrect,
                                    'Actions Accuracy': act_accuracy,
                                    'Actions Error Rate': act_error_rate,
                                    'Key Predicates Correct': predicate_correct,
                                    'Key Predicates Incorrect': predicate_incorrect,
                                    'Key Predicates Accuracy': predicate_accuracy,
                                    'Key Predicates Error Rate': predicate_error_rate,
                                    'Key Objects Correct': object_correct,
                                    'Key Objects Incorrect': object_incorrect,
                                    'Key Objects Accuracy': object_accuracy,
                                    'Key Objects Error Rate': object_error_rate}, ignore_index=True)


# 将结果保存到Excel文件
results_df.to_excel("results.xlsx", index=False)
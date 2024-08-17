import copy
import os
import matplotlib.pyplot as plt
from collections import Counter
import random
from btgym.utils import ROOT_PATH
import pandas as pd
import numpy as np
import time
import re
import btgym
from btgym.utils.tools import collect_action_nodes
from btgym.utils.read_dataset import read_dataset
from btgym.algos.llm_client.tools import goal_transfer_str
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process, act_format_records
from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction
from btgym.envs.RobotHow_Small.exec_lib._base.RHSAction import RHSAction
from btgym.envs.RoboWaiter.exec_lib._base.RWAction import RWAction
from btgym.envs.VirtualHome.exec_lib._base.VHAction import VHAction
from btgym.utils.tools import *

os.chdir(f'{ROOT_PATH}/../z_benchmark')
from btgym.algos.llm_client.llm_ask_tools import extract_llm_from_instr_goal, extract_llm_from_reflect, \
    convert_conditions, format_example
import concurrent.futures
from btgym.algos.llm_client.llms.gpt3 import LLMGPT3

llm = LLMGPT3()

scene = "VH"
difficulty = "single"

for scene in ['VH']: #['RH', 'RHS', 'RW', 'VH']
    for difficulty in ["single"]: #'multi'
        print(f"=============== {scene}  {difficulty} ====================")
        # 读入数据
        output_path = f"{ROOT_PATH}/../z_benchmark/llm_data/{scene}_{difficulty}_100_llm_data.txt"
        default_prompt_file = f"{ROOT_PATH}/../z_benchmark/llm_output_act/prompt_{scene}.txt"
        data_path = f"{ROOT_PATH}/../z_benchmark/data/{scene}_{difficulty}_100_processed_data.txt"
        data = read_dataset(data_path)

        # 遍历每条数据大模型预测并写入
        # 这里能不能并行写
        start_num=0
        data = data[start_num:]
        for i,d in enumerate(data):
            priority_act_ls, llm_key_pred, llm_key_obj, messages, distances, parsed_fail = \
                extract_llm_from_instr_goal(llm, default_prompt_file, 1, d['Goals'], verbose=False,
                                            choose_database=False)
        # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor: #max_workers=1
        #     futures = [
        #         executor.submit(extract_llm_from_instr_goal, llm, default_prompt_file, 1, d['Goals'], verbose=False,
        #                         choose_database=False)
        #         for i, d in enumerate(data)]
        #     for i, (future, d) in enumerate(zip(concurrent.futures.as_completed(futures), data)):
        #         priority_act_ls, llm_key_pred, llm_key_obj, messages, distances, parsed_fail = future.result()

            print("i+1:", i + 1 + start_num, " ", priority_act_ls, "\n")
            def extract_and_format(items):
                return ', '.join(items)
            correct_act, predicate, objects = act_format_records(priority_act_ls)
            formatted_act = extract_and_format(correct_act)
            formatted_predicates = extract_and_format(d['Vital Action Predicates'])
            formatted_objects = extract_and_format(d['Vital Objects'])

            entry_str = f"{i + 1 + start_num}\n"
            entry_str += f"Environment:1\n"
            entry_str += f"Instruction: {d['Instruction']}\n"
            entry_str += f"Goals: {' & '.join(d['Goals'])}\n"
            entry_str += f"Optimal Actions: {formatted_act}\n"
            entry_str += f"Vital Action Predicates: {formatted_predicates}\n"
            entry_str += f"Vital Objects: {formatted_objects}\n"
            write_to_file(entry_str, output_path)

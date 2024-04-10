import time
from btgym import BehaviorTree
from btgym import ExecBehaviorLibrary
import btgym
from btgym.utils import ROOT_PATH
from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from sympy import symbols, Not, Or, And, to_dnf
from sympy import symbols, simplify_logic

import csv
import re
from dataset import read_dataset

from btgym.algos.llm_client.tools import goal_transfer_str,act_str_process

env = btgym.make("VH-PutMilkInFridge")


def get_time(goal, act_ls):
    # todo: BTExp:process
    cur_cond_set=env.agents[0].condition_set = {"IsSwitchedOff(tv)","IsSwitchedOff(faucet)","IsSwitchedOff(stove)", "IsSwitchedOff(dishwasher)",
                                                "IsSwitchedOn(lightswitch)","IsSwitchedOn(tablelamp)",
                                                "IsSwitchedOff(coffeemaker)","IsSwitchedOff(toaster)","IsSwitchedOff(microwave)",
                                                "IsSwitchedOff(computer)","IsSwitchedOff(radio)",

                                                "IsClose(fridge)","IsClose(bathroomcabinet)","IsClose(stove)","IsClose(dishwasher)","IsClose(microwave)",
                                                "IsClose(toilet)",

                                   "IsRightHandEmpty(self)","IsLeftHandEmpty(self)","IsStanding(self)"
                                   }
    start_time = time.time()
    # priority_act_ls = []
    algo = BTExpInterface(env.behavior_lib, cur_cond_set, act_ls)
    ptml_string = algo.process(goal)
    end_time = time.time()
    planning_time_total = (end_time - start_time)
    print("planning_time_total:", planning_time_total)
    print("cost_total:", algo.algo.min_cost)

    return algo.algo.min_cost, planning_time_total, ptml_string

# env = btgym.make("VH-PutMilkInFridge")


prompt_file = f"{ROOT_PATH}\\algos\\llm_client\\prompt.txt"
with open(prompt_file, 'r', encoding="utf-8") as f:
    prompt = f.read().strip()
# print(prompt)

# Instruction = "Put the bowl in the dishwasher and wash it."
# Instruction="Put the milk and chicken in the fridge."
# Instruction="Turn on the computer, TV, and lights, then put the bowl in the dishwasher and wash it"
dataset = read_dataset('C:/Users/yangz/Desktop/dataset.txt')

# 打开一个文件写入,'newline' ='',避免空行
with open('example.csv', 'w', newline='') as csvfile:
    # 创建一个 csv writer 对象
    fieldnames = ['example','Instruction',
                  'Goals', 'answer','goals',
                  'Actions','actions',
                  'cost1', 'cost2',
                  'time1','time2']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    # 写入列名
    writer.writeheader()

for index, example in enumerate(dataset):

    # print(example)
    if index != 13:
        continue
    example_number = f"example {index + 1}"
    Instruction = example['Instruction']
    Goals = example['Goals']
    Actions = example['Actions']
    instruction = Instruction
    print(instruction)

    # todo: LLM
    llm = LLMGPT3()
    question = prompt + instruction
    answer = llm.request(question=question)
    # print('LLM')
    # 字符串处理
    answer = answer.replace('|', '')
    answer = answer.replace('\n[', '')
    answer = answer.replace(']\n', '')
    answer = answer.replace('*', '')
    part = answer.split("Goals:")
    answer = "Goals:".join(part[1:])
    # print(answer)
    goal_str = answer.split("Actions:")[0].replace("Goals:", "").strip()
    act_str = answer.split("Actions:")[1]
    goal_set = goal_transfer_str(goal_str)
    cost1, time1, ptml_string1 = get_time(Goals, [])
    # priority_act_ls = []
    priority_act_ls = act_str_process(act_str)  # 生成推荐动作
    cost2, time2, ptml_string2 = get_time(Goals, priority_act_ls)
    # print('priority_act_ls', priority_act_ls)
    # 打开一个文件用于写入。注意 'newline' 参数被设置为 ''，以避免在写入时出现额外的空行。
    with open('example.csv', 'a', newline='') as csvfile:
        # 逐行写入数据
        actions = ','.join(priority_act_ls)
        # print(actions2)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'example': example_number, 'Instruction': Instruction, 'Goals': Goals, 'Actions': Actions,
                         'answer': answer,
                         'goals': goal_set, 'actions': actions,
                         'cost1': cost1, 'time1': time1,
                         'cost2': cost2, 'time2': time2})
    file_name = example_number


    file_path = f'./{file_name}_1.btml'
    with open(file_path, 'w') as file:
        file.write(ptml_string1)
    file_path = f'./{file_name}_2.btml'

    with open(file_path, 'w') as file:
        file.write(ptml_string2)

    # while True:
    #     try:
    #         # todo: LLM
    #         llm = LLMGPT3()
    #         question = prompt + instruction
    #         answer = llm.request(question=question)
    #         print('LLM')
    #         # 字符串处理
    #         answer = answer.replace('|', '')
    #         answer = answer.replace('\n[', '')
    #         answer = answer.replace(']\n', '')
    #         answer = answer.replace('*', '')
    #         part = answer.split("Goals:")
    #         answer = "Goals:".join(part[1:])
    #         print(answer)
    #         goal_str = answer.split("Actions:")[0].replace("Goals:", "").strip()
    #         act_str = answer.split("Actions:")[1]
    #         goal_set = goal_transfer_str(goal_str)
    #         cost1, time1, ptml_string1 = get_time(goal_set, [])
    #         # priority_act_ls = []
    #         priority_act_ls = act_str_process(act_str)  # 生成推荐动作
    #         cost2, time2, ptml_string2 = get_time(goal_set, priority_act_ls)
    #         print('priority_act_ls', priority_act_ls)
    #         # 打开一个文件用于写入。注意 'newline' 参数被设置为 ''，以避免在写入时出现额外的空行。
    #         with open('example.csv', 'a', newline='') as csvfile:
    #             # 逐行写入数据
    #             actions2 = ','.join(priority_act_ls)
    #             # print(actions2)
    #             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #             writer.writerow({'example': example_number, 'Instruction': Instruction, 'Goals': Goals, 'Actions': Actions,
    #                              'answer': answer,
    #                              'goals1': goal_set, 'actions1': act_str,
    #                              'goals2': goal_set, 'actions2': actions2,
    #                              'cost1': cost1, 'time1': time1,
    #                              'cost2': cost1, 'time2': time2})
    #         file_name = example_number
    #
    #         file_path = f'./{file_name}_1.btml'
    #         with open(file_path, 'w') as file:
    #             file.write(ptml_string1)
    #         file_path = f'./{file_name}_2.btml'
    #
    #         with open(file_path, 'w') as file:
    #             file.write(ptml_string2)
    #         break
    #
    #     except:
    #         print('some errors occured')


# # 读取执行
# bt = BehaviorTree("example 30_1.btml", env.behavior_lib)
# bt.print()
# # bt.draw()
#
# env.agents[0].bind_bt(bt)
# env.reset()
# env.print_ticks = False
#
# is_finished = False
# while not is_finished:
#     is_finished = env.step()
#     # print(env.agents[0].condition_set)
#
#     g_finished=True
#     for g in goal_set:
#         if not g<= env.agents[0].condition_set:
#             g_finished=False
#         if g_finished:
#             is_finished=True
# env.close()
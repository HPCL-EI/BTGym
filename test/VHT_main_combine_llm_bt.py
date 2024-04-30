import time
import re
from btgym import BehaviorTree
from btgym import ExecBehaviorLibrary
import btgym
from btgym.utils import ROOT_PATH
# from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction
from btgym.algos.bt_autogen.tools import state_transition
from sympy import symbols, Not, Or, And, to_dnf
from sympy import symbols, simplify_logic
import re

from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process
import random
import numpy as np

seed = 0
random.seed(seed)
np.random.seed(seed)

# env = btgym.make("VHT-WatchTV")
env = btgym.make("VHT-PutMilkInFridge")
# print(env.graph_input['nodes'])
cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
cur_cond_set |= {f'IsClose({arg})' for arg in VHTAction.CAN_OPEN}
cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHTAction.HAS_SWITCH}
cur_cond_set |= {f'IsUnplugged({arg})' for arg in VHTAction.HAS_PLUG}



# # todo: LLMs
prompt_file = f"{ROOT_PATH}\\algos\\llm_client\\prompt_VHT.txt"
with open(prompt_file, 'r', encoding="utf-8") as f:
    prompt = f.read().strip()
# print(prompt)


# # instuction = "Put the bowl in the dishwasher and wash it."
# # instuction="Put the milk and chicken in the fridge."
# # instuction="Put the milk or chicken in the fridge."
# instuction="Turn on the computer, TV, and lights, then put the bowl in the dishwasher and wash it"
# instuction="Prepare for a small birthday party by setting the dining table with candles, plates, and wine glasses. "+\
#     "Then, bake a cake using the oven, ensure the candles are switched on."+\
#     "Finally, make sure the kitchen counter is clean."
instuction = "Prepare for a small birthday party by baking a cake using the oven, ensure the candles are switched on." + \
             "Finally, make sure the kitchen counter is clean."

# llm = LLMGPT3()
# message=[{"role": "system", "content": ""}]
# question = prompt+instuction

# 补充：向量数据库检索，拼接上最相近的 Example cur_cond_set
cur_env_state = ', '.join(map(str, cur_cond_set))
cur_data = instuction+"\n[current environmental condition]\n"+cur_env_state # 可能还要再调整
# cur_emb = llm.embedding(question=cur_data)
# 导入向量数据库，找到最近的前5条。
# 准备好的 30条数据 作为 向量数据库
example = ""
# 将例子拼在后面
# question+=example

messages = []
# messages.append({"role": "user", "content": question})
# answer = llm.request(message=messages)
# messages.append({"role": "assistant", "content": answer})

# goal_set = [{'IsIn(milk,fridge)','IsSwitchedOn(candle)'}]
# priority_act_ls = ["Walk(milk)", "RightGrab(milk)", "Walk(fridge)",'Open(fridge)',
#                    "RightPutIn(milk,fridge)",'PlugIn(fridge)', 'Walk(candle)',"SwitchOn(candle)"]
goal_set = [{'IsPlugged(fridge)'}]

priority_act_ls = ['Walk(fridge)']


priority_obj_ls = []
# 提取目标中的所有物体
objects = set()
# 正则表达式用于找到括号中的内容
pattern = re.compile(r'\((.*?)\)')
# 遍历所有表达式，提取物体名称
for expr in goal_set[0]:
    # 找到括号内的内容
    match = pattern.search(expr)
    if match:
        # 将括号内的内容按逗号分割并加入到集合中
        objects.update(match.group(1).split(','))
priority_obj_ls += list(objects)


# todo: BTExp:process
MAX_TIME = 3
for try_time in range(MAX_TIME):

    # 在小动作空间里搜索
    algo = BTExpInterface(env.behavior_lib, cur_cond_set, priority_act_ls, priority_obj_ls, \
                          selected_algorithm="opt", choose_small_action_space=True,\
                          llm_reflect=False)

    start_time = time.time()
    algo.process(goal_set)
    end_time = time.time()

    ptml_string, cost, expanded_num = algo.post_process()  # 后处理
    print("Expanded Conditions: ", expanded_num)
    planning_time_total = (end_time - start_time)
    print("planning_time_total:", planning_time_total)
    print("cost_total:", cost)
    file_name = "grasp_milk"
    file_path = f'./{file_name}.btml'
    with open(file_path, 'w') as file:
        file.write(ptml_string)

    # 读取执行
    bt = BehaviorTree(file_name + ".btml", env.behavior_lib)
    # bt.print()
    # bt.draw()

    env.agents[0].bind_bt(bt)
    env.reset()
    env.print_ticks = True

    # simulation and test
    print("\n================ ")


    goal = goal_set[0]
    state = cur_cond_set
    steps = 0
    current_cost = 0
    current_tick_time = 0

    act_num=1

    val, obj, cost, tick_time = algo.algo.bt.cost_tick(state, 0, 0)  # tick行为树，obj为所运行的行动
    print(f"Action: {act_num}  {obj.__str__().ljust(35)}cost: {cost}")
    current_tick_time += tick_time
    current_cost += cost
    while val != 'success' and val != 'failure':
        state = state_transition(state, obj)
        val, obj, cost, tick_time = algo.algo.bt.cost_tick(state, 0, 0)
        act_num+=1
        print(f"Action: {act_num}  {obj.__str__().ljust(35)}cost: {cost}")
        current_cost += cost
        current_tick_time += tick_time
        if (val == 'failure'):
            print("bt fails at step", steps)
            error = True
            break
        steps += 1
        if (steps >= 500):  # 至多运行500步
            break
    if goal <= state:  # 错误解，目标条件不在执行后状态满足
        print("Finished!")
    print(f"一定运行了 {act_num-1} 个动作步")
    print("current_cost:",current_cost)
    print("================ ")


    # 大模型反馈
    if goal <= state:
        break

    print("大模型重推荐......")
    prompt = ""
    prompt += (
            "\nThis is the action tree currently found by the reverse expansion algorithm of the behavior tree. " + \
            "To reach the goal state, please recommend the key actions needed next to reach the goal state based on the existing action tree. " + \
            "This time, there is no need to output the goal state, only the key actions are needed. " + \
            'The format for presenting key actions should start with the word \'Actions:\'. ')


    # messages.append({"role": "user", "content": prompt})
    # answer = llm.request(message=messages)
    # messages.append({"role": "assistant", "content": answer})
    # print("answer:",answer)
    # act_str = answer.split("Actions:")[1]
    # # act_str = re.sub(r'\s+|[\[\]\(\)\n]', '', act_str)
    # # priority_act_ls = act_str_process(act_str)
    # priority_act_ls = [action.replace(" ", "") for action in act_str.split(",")]
    # print(priority_act_ls)

    # priority_act_ls, priority_obj_ls
    # 问题：小动作空间，是不是不是分级启发式，priority_obj_ls限制动作空间，priority_act_ls采用计数的启发式
    # 问题：大模型重新推荐，是不是全部都要重新推荐


# env.close()



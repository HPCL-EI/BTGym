import time
import re
from btgym import BehaviorTree
from btgym import ExecBehaviorLibrary
import btgym
from btgym.utils import ROOT_PATH
from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction

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

# # todo: LLMs
prompt_file = f"{ROOT_PATH}\\algos\\llm_client\\prompt_VHT.txt"
with open(prompt_file, 'r', encoding="utf-8") as f:
    prompt = f.read().strip()
# print(prompt)
#
# # instuction = "Put the bowl in the dishwasher and wash it."
# # instuction="Put the milk and chicken in the fridge."
# # instuction="Put the milk or chicken in the fridge."
# instuction="Turn on the computer, TV, and lights, then put the bowl in the dishwasher and wash it"
# instuction="Prepare for a small birthday party by setting the dining table with candles, plates, and wine glasses. "+\
#     "Then, bake a cake using the oven, ensure the candles are switched on."+\
#     "Finally, make sure the kitchen counter is clean."
instuction = "Prepare for a small birthday party by baking a cake using the oven, ensure the candles are switched on." + \
             "Finally, make sure the kitchen counter is clean."

llm = LLMGPT3()
# message=[{"role": "system", "content": ""}]
# question = prompt+instuction
messages = []
# messages.append({"role": "user", "content": question})
# answer = llm.request(message=messages)
# messages.append({"role": "assistant", "content": answer})
#
# print(answer)
# goal_str = answer.split("Actions:")[0].replace("Goals:", "").strip()
# act_str = answer.split("Actions:")[1]
# goal_set = goal_transfer_str(goal_str)
# priority_act_ls = act_str_process(act_str)
# print("goal",goal_set)
# print("act:",priority_act_ls)

# goal_set = [
#     {'IsIn(poundcake,fridge)','IsClean(kitchencounter)'}
# ]
# priority_act_ls = [
#     "Walk(poundcake)", "RightGrab(poundcake)", "Walk(fridge)","PlugIn(fridge)", "Open(fridge)","RightPutIn(poundcake,fridge)",
#     "Walk(rag)", "RightGrab(rag)", "Walk(kitchencounter)", "Wipe(kitchencounter)"
# ]


# goal_set = [
#     {'IsIn(poundcake,microwave)', 'IsSwitchedOn(microwave)',
#      'IsClean(kitchencounter)'
#      }
# ]
# priority_act_ls = [
#     "Walk(poundcake)", "RightGrab(poundcake)", "Walk(microwave)", "Open(microwave)","RightPutIn(poundcake,microwave)",
#     "PlugIn(microwave)","Close(microwave)","SwitchOn(microwave)",
#     "Walk(rag)", "RightGrab(rag)", "Walk(kitchencounter)", "Wipe(kitchencounter)"
# ]



# goal_set = [
#     {'IsIn(poundcake,oven)', 'IsSwitchedOn(oven)', 'IsClean(kitchencounter)', 'IsSwitchedOn(candle)'}
# ]
# priority_act_ls = [
#     "Walk(poundcake)", "RightGrab(poundcake)", "Walk(oven)", "PlugIn(oven)", "Open(oven)","RightPutIn(poundcake,oven)",
#     "SwitchOn(oven)", #"Close(oven)",
#     "Walk(candle)", "SwitchOn(candle)",
#     "Walk(rag)", "RightGrab(rag)", "Walk(kitchencounter)", "Wipe(kitchencounter)"
# ]

goal_set = [
    {'IsOn(candle,kitchentable)', 'IsOn(plate,kitchentable)', 'IsOn(wineglass,kitchentable)',
    'IsIn(poundcake,oven)','IsSwitchedOn(oven)', 'IsSwitchedOn(candle)', 'IsClean(kitchencounter)'}
]
priority_act_ls=[
    "Walk(candle)", "RightGrab(candle)", "Walk(plate)", "LeftGrab(plate)",
    "Walk(kitchentable)", "RightPut(candle,kitchentable)", "LeftPut(plate,kitchentable)",
    "Walk(wineglass)", "RightGrab(wineglass)", "Walk(kitchentable)", "RightPut(wineglass,kitchentable)",
    "Walk(poundcake)", "RightGrab(poundcake)", "Walk(oven)", "PlugIn(oven)", "Open(oven)", "RightPutIn(poundcake,oven)",
    "Walk(candle)", "SwitchOn(candle)",
    "Walk(rag)", "RightGrab(rag)", "Walk(kitchencounter)", "Wipe(kitchencounter)"
]


# 冰箱放入东西前要插上电
# goal_set = [{'IsClose(fridge)', 'IsIn(milk,fridge)', 'IsIn(chicken,fridge)'}]
# priority_act_ls = ["Walk(milk)","RightGrab(milk)","Walk(fridge)","Open(fridge)","RightPutIn(milk,fridge)","Walk(chicken)",
#                    "LeftGrab(chicken)","LeftPutIn(chicken,fridge)",'PlugIn(fridge)']
# priority_obj_ls = []

# goal_set = [{'IsClean(bench)','IsClean(sofa)','IsClean(dishwasher)'}]
# priority_act_ls = ["Walk(brush)","RightGrab(brush)","Walk(bench)","Wipe(bench)",
#                    "Walk(sofa)","Wipe(sofa)","Walk(dishwasher)","Wipe(dishwasher)"]
# priority_obj_ls = ["brush"]

# goal_set = [{'IsIn(milk,fridge)'}]
# priority_act_ls = ["Walk(milk)", "RightGrab(milk)", "Walk(fridge)", "Open(fridge)", "RightPutIn(milk,fridge)",'PlugIn(fridge)'] #,

# goal_set = [{'IsIn(milk,microwave)','IsSwitchedOn(microwave)'}]
# priority_act_ls = ["Walk(milk)", "RightGrab(milk)", "Walk(microwave)", "Open(microwave)", "RightPutIn(milk,fridge)",'PlugIn(microwave)','SwitchOn(microwave)'] #,

# goal_set = [{'IsIn(milk,microwave)','IsSwitchedOn(microwave)'}]
# priority_act_ls = [ "Walk(milk)", "RightGrab(milk)", "Walk(microwave)", "Open(microwave)", \
#                     "RightPutIn(milk,microwave)",'PlugIn(microwave)','SwitchOn(microwave)'] #,

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
# priority_obj_ls += list(objects)


# todo: BTExp:process
cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
cur_cond_set |= {f'IsClose({arg})' for arg in VHTAction.CAN_OPEN}
cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHTAction.HAS_SWITCH}
cur_cond_set |= {f'IsUnplugged({arg})' for arg in VHTAction.HAS_PLUG}

algo = BTExpInterface(env.behavior_lib, cur_cond_set, priority_act_ls, priority_obj_ls, \
                      selected_algorithm="opt", llm_reflect=True, llm=llm, messages=messages)

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
env.print_ticks = False

# simulation and test
print("\n================ ")
from btgym.algos.bt_autogen.tools import state_transition

goal = goal_set[0]
state = cur_cond_set
steps = 0
current_cost = 0
current_tick_time = 0

val, obj, cost, tick_time = algo.algo.bt.cost_tick(state, 0, 0)  # tick行为树，obj为所运行的行动
print("Action:  ", obj)
current_tick_time += tick_time
current_cost += cost
while val != 'success' and val != 'failure':
    state = state_transition(state, obj)
    val, obj, cost, tick_time = algo.algo.bt.cost_tick(state, 0, 0)
    print("Action:  ", obj)
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
print("================ ")
env.close()

# is_finished = False
# while not is_finished:
#     is_finished = env.step()
#     # print(env.agents[0].condition_set)
#
#     g_finished = True
#     for g in goal_set:
#         if not g <= env.agents[0].condition_set:
#             g_finished = False
#         if g_finished:
#             is_finished = True
# env.close()
# print("\n====== batch scripts ======\n")

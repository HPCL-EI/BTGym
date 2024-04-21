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
seed=0
random.seed(seed)
np.random.seed(seed)


# env = btgym.make("VHT-WatchTV")
env = btgym.make("VHT-PutMilkInFridge")
# print(env.graph_input['nodes'])

# # todo: LLMs
# prompt_file = f"{ROOT_PATH}\\algos\\llm_client\\prompt.txt"
# with open(prompt_file, 'r', encoding="utf-8") as f:
#     prompt = f.read().strip()
# # print(prompt)
#
# # instuction = "Put the bowl in the dishwasher and wash it."
# # instuction="Put the milk and chicken in the fridge."
# # instuction="Put the milk or chicken in the fridge."
# instuction="Turn on the computer, TV, and lights, then put the bowl in the dishwasher and wash it"
#
# llm = LLMGPT3()
# question = prompt+instuction
# answer = llm.request(question=question)
# print(answer)
#
# goal_str = answer.split("Actions:")[0].replace("Goals:", "").strip()
# act_str = answer.split("Actions:")[1]
#
# goal_set = goal_transfer_str(goal_str)
# priority_act_ls = act_str_process(act_str)
#
# print("goal",goal_set)
# print("act:",priority_act_ls)


# goal_set = [{'IsClose(fridge)', 'IsIn(milk,fridge)', 'IsIn(chicken,fridge)'}]
# goal_set = [{'IsIn(milk,fridge)'}]
# goal_set = [{'IsOn(milk,sofa)'}]
# goal_set = [{'IsOpen(fridge)'}]
# priority_act_ls = ["Walk(milk)", "RightGrab(milk)", "Walk(fridge)", "Open(fridge)", "RightPutIn(milk,fridge)"]
# priority_act_ls = ["Walk(milk)","RightGrab(milk)","Walk(fridge)","Open(fridge)","RightPutIn(milk,fridge)","Walk(chicken)",
#                    "LeftGrab(chicken)","LeftPutIn(chicken,fridge)"]
# priority_act_ls=[]

# 冰箱放入东西前要插上电
# goal_set = [{'IsClose(fridge)', 'IsIn(milk,fridge)', 'IsIn(chicken,fridge)'}]
# priority_act_ls = ["Walk(milk)","RightGrab(milk)","Walk(fridge)","Open(fridge)","RightPutIn(milk,fridge)","Walk(chicken)",
#                    "LeftGrab(chicken)","LeftPutIn(chicken,fridge)",'PlugIn(fridge)']
# priority_obj_ls = []

# goal_set = [{'IsClean(fridge)','IsClean(tv)','IsClean(kitchentable)','IsClean(bench)','IsClean(sofa)','IsClean(dishwasher)'}]
# priority_act_ls = ["Walk(brush)","RightGrab(brush)","Walk(bench)","Wipe(bench)",
#                    "Walk(sofa)","Wipe(sofa)","Walk(dishwasher)"]
# priority_obj_ls = ["brush"]

goal_set = [{'IsIn(milk,fridge)'}]
priority_act_ls = ["Walk(milk)", "RightGrab(milk)", "Walk(fridge)", "Open(fridge)", "RightPutIn(milk,fridge)",'PlugIn(fridge)'] #,

# goal_set = [{'IsWatching(self,tv)'}]
# priority_act_ls=['Walk(tv)','Watch(tv)']

# goal_set = [{'IsIn(chips,fridge)'}]
# priority_act_ls = ["Walk(chips)", "RightGrab(chips)", "Walk(fridge)", "Open(fridge)", "RightPutIn(chips,fridge)"]

# goal_set = [{'IsClean(tv)'}]
# priority_act_ls=['Walk(papertowel)','RightGrab(papertowel)','RightGrab(tv)',"Wipe(tv)"]

# goal_set = [{'IsSwitchedOn(tv)'}]
# priority_act_ls=['Walk(tv)','PlugIn(tv)','SwitchOn(tv)']

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
cur_cond_set = env.agents[0].condition_set = {"IsSwitchedOff(tv)", "IsSwitchedOff(faucet)", "IsSwitchedOff(stove)",
                                              "IsSwitchedOff(dishwasher)",
                                              "IsSwitchedOn(lightswitch)", "IsSwitchedOn(tablelamp)",
                                              "IsSwitchedOff(coffeemaker)", "IsSwitchedOff(toaster)",
                                              "IsSwitchedOff(microwave)",
                                              "IsSwitchedOff(computer)", "IsSwitchedOff(radio)",

                                              "IsClose(fridge)", "IsClose(bathroomcabinet)", "IsClose(stove)",
                                              "IsClose(dishwasher)", "IsClose(microwave)",
                                              "IsClose(toilet)",

                                              "IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"
                                              }
cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHTAction.HAS_SWITCH}
cur_cond_set |= {f'IsUnplugged({arg})' for arg in VHTAction.HAS_PLUG}



algo = BTExpInterface(env.behavior_lib, cur_cond_set, priority_act_ls, priority_obj_ls,\
                      selected_algorithm="opt",llm_reflect=True)

start_time = time.time()
algo.process(goal_set)
end_time = time.time()

ptml_string, cost, expanded_num = algo.post_process()  # 后处理
print("Expanded Conditions: ",expanded_num)
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

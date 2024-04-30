import time

from btgym import BehaviorTree
from btgym import ExecBehaviorLibrary
import btgym
from btgym.utils import ROOT_PATH
from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
from btgym.algos.bt_autogen.main_interface import BTExpInterface,collect_action_nodes
from btgym.envs.virtualhometext.exec_lib._base.VHTAction_small import VHTAction_small
from sympy import symbols, Not, Or, And, to_dnf
from sympy import symbols, simplify_logic
import re
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process
from btgym.algos.bt_autogen.Action import Action
import random
import numpy as np
seed=0
random.seed(seed)
np.random.seed(seed)

from tools import collect_action_nodes


env = btgym.make("VHT-Small")
action_list = collect_action_nodes(env.behavior_lib)
all_actions_set = set(action_list)

# Easy
# goal_set = [{'IsClean(nightstand)'}]
# true_priority_act_set = {'Walk(rag)','RightGrab(rag)','Walk(nightstand)','Wipe(nightstand)'}

# Medium
goal_set = [{'IsClean(nightstand)','IsOn(clock,nightstand)'}]
true_priority_act_set = {'Walk(rag)','RightGrab(rag)','Walk(clock)','LeftGrab(clock)',\
                         'Walk(nightstand)','Wipe(nightstand)','LeftPut(clock,nightstand)'}

# Hard
# goal_set = [{'IsIn(milk,microwave)','IsSwitchedOn(microwave)'}]
# true_priority_act_set = {"Walk(milk)", "RightGrab(milk)", "Walk(microwave)", "Open(microwave)","PlugIn(microwave)", \
#                     "RightPutIn(milk,microwave)",'SwitchOn(microwave)'}

error_priority_act_set = all_actions_set-true_priority_act_set
# 推荐优先级

priority_act_ls=set()
error_rate = 0.5
correct_rate = 0.5

error_num=int(len(true_priority_act_set)*error_rate)
correct_num=int(len(true_priority_act_set)*correct_rate)

priority_act_ls |= set(random.sample(list(error_priority_act_set), error_num))
priority_act_ls |= set(random.sample(list(true_priority_act_set), correct_num))

print(priority_act_ls)


cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
cur_cond_set |= {f'IsClose({arg})' for arg in VHTAction_small.CAN_OPEN}
cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHTAction_small.HAS_SWITCH}
cur_cond_set |= {f'IsUnplugged({arg})' for arg in VHTAction_small.HAS_PLUG}


priority_obj_ls=[]
# 这里写错了 baseline 应该是不推荐的
# algo = BTExpInterface(None, cur_cond_set, priority_act_ls, priority_obj_ls,\
#                       selected_algorithm="baseline",action_list=action_list)

# 这里应该是推荐错了怎么处理
algo = BTExpInterface(None, cur_cond_set, priority_act_ls, priority_obj_ls,\
                      selected_algorithm="opt",action_list=action_list)

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
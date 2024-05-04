import time

from btgym import BehaviorTree
from btgym import ExecBehaviorLibrary
import btgym
from btgym.utils import ROOT_PATH
from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
from btgym.algos.bt_autogen.main_interface import BTExpInterface,collect_action_nodes
from btgym.envs.virtualhometext.exec_lib._base.VHTAction_small import VHTAction_small
from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction
from btgym.algos.bt_autogen.tools import state_transition
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
from read_dataset import read_dataset,read_environment

# env_path = f"{ROOT_PATH}/../test/dataset/environment.txt"
# env_dic = read_environment(env_path,style=True)

#########################


import pickle

# def refresh_VHT_samll_data():
# 读入数据集合
# data_path = f"{ROOT_PATH}/../test/dataset/data0429.txt"
# data = read_dataset(data_path)
data_path = f"{ROOT_PATH}/../test/dataset/dataset_noenv.txt"
data = read_dataset(data_path)

# data = data[:10]

data_num = len(data)
print(f"导入 {data_num} 条数据")
print(data[0])

# 数据集中涉及的所有物体集合
objs=set()
for d in data:
    objs |= set(d['Key_Object'])

categories = ['SURFACES', 'SITTABLE', 'CAN_OPEN', 'CONTAINERS', 'GRABBABLE', 'cleaning_tools', \
         'cutting_tools', 'HAS_SWITCH', 'HAS_PLUG', 'CUTABLE', 'EATABLE', 'WASHABLE', 'RECIPIENT', \
         'POURABLE', 'DRINKABLE']
categories_objs_dic={}
for ctg in categories:
    categories_objs_dic[ctg] = getattr(VHTAction, ctg)
    categories_objs_dic[ctg] &= objs


ctg_objs_path = f"{ROOT_PATH}/../test/EXP/ctg_objs.pickle"
# 打开一个文件用于写入，注意'b'表示二进制模式
with open(ctg_objs_path, 'wb') as file:
    # 使用pickle.dump()函数将数据写入文件
    pickle.dump(categories_objs_dic, file)
################






env = btgym.make("VHT-Small")
action_list = collect_action_nodes(env.behavior_lib)
all_actions_set = set(action_list)
all_actions_str_set = {act.name for act in all_actions_set}
# Easy
# goal_set = [{'IsClean(nightstand)'}]
# true_priority_act_set = {'Walk(rag)','RightGrab(rag)','Walk(nightstand)','Wipe(nightstand)'}

# Medium
# 正确的动作在数据集外面
# goal_set = [{'IsClean(nightstand)','IsOn(clock,nightstand)'}]
# true_priority_act_set = {'Walk(rag)','RightGrab(rag)','Walk(clock)','LeftGrab(clock)',\
#                          'Walk(nightstand)','Wipe(nightstand)','LeftPut(clock,nightstand)'}
# goal_set = [{'IsClean(bed)','IsOn(toy,bed)'}]
# true_priority_act_set = {'Walk(rag)','RightGrab(rag)','Walk(toy)','LeftGrab(toy)',\
#                          'Walk(bed)','Wipe(bed)','LeftPut(toy,bed)'}

# Hard
# goal_set = [{'IsIn(milk,microwave)','IsSwitchedOn(microwave)'}]
# true_priority_act_set = {"Walk(milk)", "RightGrab(milk)", "Walk(microwave)", "Open(microwave)","PlugIn(microwave)", \
#                     "RightPutIn(milk,microwave)",'SwitchOn(microwave)'}
# goal_set = [{'IsIn(milk,microwave)','IsSwitchedOn(microwave)'}]
# true_priority_act_set = {"Walk(milk)", "RightGrab(milk)", "Walk(microwave)", "Open(microwave)","PlugIn(microwave)", \
#                     "RightPutIn(milk,microwave)",'SwitchOn(microwave)'}


combined_string = ' & '.join(data[0]['Goals'])
goal_set = goal_transfer_str(combined_string)
true_priority_act_set = set(act_str_process(data[0]['Actions'],already_split=True))
print("goal:",goal_set)
print("act:",true_priority_act_set)

error_priority_act_set = all_actions_str_set-true_priority_act_set
# 推荐优先级

priority_act_ls=set()
error_rate = 0.5
correct_rate = 0.5

#   0     1          0  6       7   0.0241
#  0.2  0.8          1  5      813  2.08544
#  0.5  0.5          3  3     1579  4.50097
#  0.8  0.2          4  2     1855  5.394
#  0.9  0.1          5  1     2318  8.324
#   1    0           6  0     1831  6.0884

error_num=int(len(true_priority_act_set)*error_rate)
correct_num=len(true_priority_act_set)-error_num
print("Error:",error_num)
print("Correct:",correct_num)

# 排序是为了取消随机性
priority_act_ls |= set(random.sample(sorted(list(error_priority_act_set)), error_num))
priority_act_ls |= set(random.sample(sorted(list(true_priority_act_set)), correct_num))
priority_act_ls = sorted(list(priority_act_ls))

print("priority_act_ls:",priority_act_ls)
# {'LeftPut(cupcake,floor)', 'Wipe(juice)', 'PlugIn(toaster)', 'LeftPut(towel,radio)', 'Walk(bathroomcabinet)', 'LeftPut(bellpepper,kitchencounter)'}
# priority_act_ls = {'RightGrab(pillow)', 'LeftPut(cuttingboard,kitchentable)', 'LeftPut(cuttingboard,sofa)', 'RightPut(boardgame,sofa)', 'LeftPut(cutleryknife,desk)', 'Walk(book)'}
# priority_act_ls = []

cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
cur_cond_set |= {f'IsClose({arg})' for arg in VHTAction_small.CAN_OPEN}
cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHTAction_small.HAS_SWITCH}
cur_cond_set |= {f'IsUnplugged({arg})' for arg in VHTAction_small.HAS_PLUG}

# cur_cond_set = set(env_dic[data[0]['Environment']])
# print("cur_cond_set:",cur_cond_set)




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

act_num = 1

val, obj, cost, tick_time = algo.algo.bt.cost_tick(state, 0, 0)  # tick行为树，obj为所运行的行动
print(f"Action: {act_num}  {obj.__str__().ljust(35)}cost: {cost}")
current_tick_time += tick_time
current_cost += cost
while val != 'success' and val != 'failure':
    state = state_transition(state, obj)
    val, obj, cost, tick_time = algo.algo.bt.cost_tick(state, 0, 0)
    act_num += 1
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
print(f"一定运行了 {act_num - 1} 个动作步")
print("current_cost:", current_cost)
print("================ ")

# 输出每条数据的
# 对称   0  0.3  0.5  0.7  1
# 数量
# 扩展节点数 时间
# 每条数据有5条这样的，然后保存为 csv

# 求出平均的一个结果，形成一张表
# 0    正确数 错误数  平均总数 expanded_num  time
# 0.3
# 0.5
# 0.7
# 1

# 分开算情况下
# 错误率占比0，正确从0-1
# 错误率占比0.3，正确从0-1
# 错误率占比0，5正确从0-1
# 得到结果
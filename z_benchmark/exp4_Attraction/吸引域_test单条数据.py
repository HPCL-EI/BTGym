import copy
import os
import random
from btgym.utils import ROOT_PATH
os.chdir(f'{ROOT_PATH}/../z_benchmark')
from tools import *
import time
import re
import btgym
from btgym.utils.tools import collect_action_nodes
from btgym.utils.read_dataset import read_dataset
from btgym.algos.llm_client.tools import goal_transfer_str
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process, act_format_records

from btgym.envs.RoboWaiter.exec_lib._base.RWAction import RWAction
from btgym.envs.VirtualHome.exec_lib._base.VHAction import VHAction
from btgym.envs.RobotHow_Small.exec_lib._base.RHSAction import RHSAction
from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction

# 导入环境
# scene="VH"
scene="VH"
env,cur_cond_set = setup_environment(scene)

# 导入数据
data_path = f"{ROOT_PATH}/../z_benchmark/data/{scene}_single_100_processed_data.txt"
data1 = read_dataset(data_path)

d = data1[0]
# goals = "IsClean_bananas & IsCut_bananas & IsIn_bananas_fridge"
goal_str = ' & '.join(d["Goals"])
# goal_str="IsOn_pear_kitchentable & IsOn_bananas_kitchentable"
goal_set = goal_transfer_str(goal_str)
d['Optimal Actions'] = act_str_process(d['Optimal Actions'], already_split=True)
print("Optimal Actions:", d['Optimal Actions'])

# 选择算法计算得到行为树
algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                      priority_act_ls=d['Optimal Actions'] , key_predicates=[],
                      key_objects=[],
                      selected_algorithm="bfs", mode="big",
                      llm_reflect=False, time_limit=3,
                          heuristic_choice=-1,output_just_best=False)

goal_set = goal_transfer_str(goal_str)

start_time = time.time()
algo.process(goal_set)
end_time = time.time()
planning_time_total = end_time - start_time

time_limit_exceeded = algo.algo.time_limit_exceeded

ptml_string, cost, expanded_num = algo.post_process()
error, state, act_num, current_cost, record_act_ls, ticks = algo.execute_bt(goal_set[0], cur_cond_set, verbose=False)

print(f"\x1b[32m Goal:{goal_str} \n Executed {act_num} action steps\x1b[0m",
      "\x1b[31mERROR\x1b[0m" if error else "",
      "\x1b[31mTIMEOUT\x1b[0m" if time_limit_exceeded else "")
print("current_cost:", current_cost, "expanded_num:", expanded_num, "planning_time_total:", planning_time_total, "ticks:",ticks)


# 跑算法
# 随机生成100个初始状态，看哪个能达到目标
# 提取出obj
objects=[]
pattern = re.compile(r'\((.*?)\)')
for expr in goal_set[0]:
    match = pattern.search(expr)
    if match:
        objects.append(match.group(1).split(','))
# obj_canGrab = objects[0]



successful_executions = 0  # 用于跟踪成功（非错误）的执行次数
exe_times = 1 #100
for i in range(exe_times):
    print("----------")
    print("i:",i)
    new_cur_state = modify_condition_set(VHAction, cur_cond_set)
    error, state, act_num, current_cost, record_act_ls, ticks = algo.execute_bt(goal_set[0], new_cur_state, verbose=False)
    # 检查是否有错误，如果没有，则增加成功计数
    if not error:
        successful_executions += 1
    print("----------")
# 计算非错误的执行占比
success_ratio = successful_executions / exe_times
print("成功的执行占比（非错误）: {:.2%}".format(success_ratio))
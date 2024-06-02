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
from btgym.envs.virtualhome.exec_lib._base.VHAction import VHAction
from btgym.envs.RobotHow_Small.exec_lib._base.RHSAction import RHSAction
from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction

# 导入环境
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
                      priority_act_ls=[], key_predicates=[],
                      key_objects=[],
                      selected_algorithm="opt", mode="big",
                      llm_reflect=False, time_limit=15,
                          heuristic_choice=-1,output_just_best=False)

goal_set = goal_transfer_str(goal_str)

start_time = time.time()
algo.process(goal_set)
end_time = time.time()
planning_time_total = end_time - start_time

time_limit_exceeded = algo.algo.time_limit_exceeded

ptml_string, cost, expanded_num = algo.post_process()
error, state, act_num, current_cost, record_act_ls = algo.execute_bt(goal_set[0], cur_cond_set, verbose=False)

print(f"\x1b[32m Goal:{goal_str} \n Executed {act_num} action steps\x1b[0m",
      "\x1b[31mERROR\x1b[0m" if error else "",
      "\x1b[31mTIMEOUT\x1b[0m" if time_limit_exceeded else "")
print("current_cost:", current_cost, "expanded_num:", expanded_num, "planning_time_total:", planning_time_total)


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

def modify_condition_set(cur_cond_set):
    new_cur_state = copy.deepcopy(cur_cond_set)

    # 改变位置
    near_conditions = [cond for cond in new_cur_state if cond.startswith('IsNear')]
    if near_conditions:
        # 已有位置，随机决定是改变还是移除
        if random.choice([True, False]):
            new_cur_state.remove(random.choice(near_conditions))
        new_position = random.choice(list(VHAction.AllObject))
        new_cur_state.add(f'IsNear(self,{new_position})')
    else:
        # 没有位置，添加一个
        new_position = random.choice(list(VHAction.AllObject))
        new_cur_state.add(f'IsNear(self,{new_position})')


    # 改变手上的状态，如果拿着东西，给它移到别的地方
    # 随机决定每只手的操作
    for hand in ['Left', 'Right']:
        holding_condition = next((cond for cond in new_cur_state if f'Is{hand}Holding' in cond), None)
        if holding_condition:
            if random.choice([True, False]):
                # 放下物体并将手标记为空
                new_cur_state.discard(holding_condition)
                new_cur_state.add(f'Is{hand}HandEmpty(self)')

                # 根据物体类型选择放置位置
                item = holding_condition.split(',')[1].strip().strip(')')
                # 随机决定放置在表面还是可放入位置
                if random.choice([True, False]):  # True for Surface, False for CanPutIn
                    put_position = random.choice(list(VHAction.SurfacePlaces))
                    new_cur_state.add(f'IsOn({item},{put_position})')
                else:
                    put_position = random.choice(list(VHAction.CanPutInPlaces))
                    new_cur_state.add(f'IsIn({item},{put_position})')

            else:
                # 换一个物体
                new_cur_state.discard(holding_condition)
                new_object = random.choice(list(VHAction.Objects))
                new_cur_state.add(f'Is{hand}Holding(self,{new_object})')
                new_cur_state.discard(f'Is{hand}HandEmpty(self)')
        else:
            if random.choice([True, False]):
                new_object = random.choice(list(VHAction.Objects))
                new_cur_state.add(f'Is{hand}Holding(self,{new_object})')
                new_cur_state.discard(f'Is{hand}HandEmpty(self)')


    # 随机改变 物体的固有属性
    # 处理可开关的物体（如门、窗、设备等）
    for obj in VHAction.HasSwitchObjects:
        state = 'IsSwitchedOn' if random.choice([True, False]) else 'IsSwitchedOff'
        opposite_state = 'IsSwitchedOff' if state == 'IsSwitchedOn' else 'IsSwitchedOn'
        # 添加新状态前删除相反的状态
        new_cur_state.discard(f'{opposite_state}({obj})')
        new_cur_state.add(f'{state}({obj})')

    # 处理可开关或靠近的物体（如柜子、抽屉等）
    for obj in VHAction.CanOpenPlaces:
        state = 'IsOpen' if random.choice([True, False]) else 'IsClose'
        opposite_state = 'IsClose' if state == 'IsOpen' else 'IsOpen'
        # 添加新状态前删除相反的状态
        new_cur_state.discard(f'{opposite_state}({obj})')
        new_cur_state.add(f'{state}({obj})')

    return new_cur_state


successful_executions = 0  # 用于跟踪成功（非错误）的执行次数
exe_times = 5 #100
for i in range(exe_times):
    print("----------")
    new_cur_state = modify_condition_set(cur_cond_set)
    error, state, act_num, current_cost, record_act_ls = algo.execute_bt(goal_set[0], new_cur_state, verbose=True)
    # 检查是否有错误，如果没有，则增加成功计数
    if not error:
        successful_executions += 1
    print("----------")
# 计算非错误的执行占比
success_ratio = successful_executions / exe_times
print("成功的执行占比（非错误）: {:.2%}".format(success_ratio))
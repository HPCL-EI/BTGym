import time
import re
from btgym import BehaviorTree
from btgym import ExecBehaviorLibrary
import btgym
from btgym.utils import ROOT_PATH
from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction
from itertools import chain

from sympy import symbols, Not, Or, And, to_dnf
from sympy import symbols, simplify_logic
import re
from btgym.utils.read_dataset import read_dataset
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process,act_format_records
import random
import numpy as np

seed = 0
random.seed(seed)
np.random.seed(seed)

# 使用正则表达式从谓词中抽取宾语
# 从谓词中抽取所有的宾语
def extract_objects(actions):
    # 匹配所有符合谓词(方法)的动作，例如 Walk(...)，RightGrab(...)
    pattern = re.compile(r'\w+\(([^)]+)\)')
    objects = []
    for action in actions:
        match = pattern.search(action)
        if match:
            objects.append(match.group(1))
    return objects



env = btgym.make("VHT-PutMilkInFridge")

cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
cur_cond_set |= {f'IsClose({arg})' for arg in VHTAction.CAN_OPEN}
cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHTAction.HAS_SWITCH}
cur_cond_set |= {f'IsUnplugged({arg})' for arg in VHTAction.HAS_PLUG}



# # 读入数据集
# data_path = f"{ROOT_PATH}/../test/dataset/data0506/dataset0506_revby_cys.txt"
# data_path = f"{ROOT_PATH}/../test/dataset/data0507/dataset0507.txt"
# data = read_dataset(data_path)
# len_data = len(data)
# print(f"导入 {len_data} 条数据")
# print(data[0])
#
# # 挑选出 env=1 的数据进行测试，总共 40条
# data1 = [d for d in data if d['Environment'] == 1]
# len_data = len(data1)
# print(f"环境为1的数据总共有 {len_data} 条")


# 直接读入 env=1 的数据
data_path = f"{ROOT_PATH}/../test/dataset/data1_env1_40.txt"
data1 = read_dataset(data_path)
len_data = len(data1)
print(f"导入 {len_data} 条数据")
print(data1[0])

for id,d in enumerate(data1[:]):
    print("id:",id,d["Instruction"])

    goal_str = ' & '.join(d["Goals"])
    act_str= ', '.join(d["Actions"])
    # goal_str = "IsOn_pillow_bed & IsOn_coffeepot_coffeetable & IsOn_towel_towelrack"
    # act_str=  "Walk_coffeetable, RightGrab_coffeepot, Walk_bed, RightPut_pillow_bed, RightGrab_pillow, Walk_coffeepot, RightPut_towel_towelrack, Walk_towel, Walk_pillow, Walk_towelrack, RightPut_coffeepot_coffeetable, RightGrab_towel"

    goal_set = goal_transfer_str(goal_str)
    print("goal_set:",goal_set)
    priority_act_ls = act_str_process(act_str)
    print("priority_act_ls:",priority_act_ls)

    # goal_set = [{'IsIn(milk,fridge)','IsSwitchedOn(candle)'}]
    # priority_act_ls = ["Walk(milk)", "RightGrab(milk)", "Walk(fridge)",'Open(fridge)',
    #                    "RightPutIn(milk,fridge)",'PlugIn(fridge)', 'Walk(candle)',"SwitchOn(candle)"]

    key_predicates=extract_objects(priority_act_ls)

    priority_obj_ls = []
    # 提取目标中的所有物体
    objects = set()
    # 正则表达式用于找到括号中的内容
    pattern = re.compile(r'\((.*?)\)')
    # 遍历所有表达式，提取物体名称
    for expr in chain(goal_set[0], priority_act_ls):
        # 找到括号内的内容
        match = pattern.search(expr)
        if match:
            # 将括号内的内容按逗号分割并加入到集合中
            objects.update(match.group(1).split(','))
    priority_obj_ls += list(objects)

    # algo = BTExpInterface(env.behavior_lib, cur_cond_set, priority_act_ls=priority_act_ls, key_predicates=key_predicates,key_objects=priority_obj_ls, \
    #                       selected_algorithm="opt",mode="small-predicate-objs")

    algo = BTExpInterface(env.behavior_lib, cur_cond_set, priority_act_ls=priority_act_ls, key_objects=priority_obj_ls, \
                          selected_algorithm="opt", mode="small-objs")


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

    env.agents[0].bind_bt(bt)
    env.reset()
    env.print_ticks = True

    # simulation and test
    print("\n================ ")
    from btgym.algos.bt_autogen.tools import state_transition

    goal = goal_set[0]
    state = cur_cond_set
    steps = 0
    current_cost = 0
    current_tick_time = 0

    act_num=1
    record_act = []

    val, obj, cost, tick_time = algo.algo.bt.cost_tick(state, 0, 0)  # tick行为树，obj为所运行的行动
    print(f"Action: {act_num}  {obj.__str__().ljust(35)}cost: {cost}")
    record_act.append(obj.__str__())
    current_tick_time += tick_time
    current_cost += cost
    while val != 'success' and val != 'failure':
        state = state_transition(state, obj)
        val, obj, cost, tick_time = algo.algo.bt.cost_tick(state, 0, 0)
        act_num+=1
        print(f"Action: {act_num}  {obj.__str__().ljust(35)}cost: {cost}")
        record_act.append(obj.__str__())
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
    print("================ \n")

    # 输出结果：
    record_act = record_act[:-1]
    correct_act,predicate,objects = act_format_records(record_act)

    # 函数来提取并格式化输出
    def extract_and_format(items):
        return ', '.join(items)
    # 调用函数并打印结果
    formatted_act = extract_and_format(correct_act)
    formatted_predicates = extract_and_format(predicate)
    formatted_objects = extract_and_format(objects)

    print("Goals:",goal_str)
    print("Actions:",formatted_act)
    print("Vital Action Predicates:",formatted_predicates)
    print("Vital Objects:",formatted_objects)

    priority_act = set(act_str.replace(" ", "").split(","))
    print("增加了：",set(correct_act)-priority_act)
    print("减少了：",priority_act-set(correct_act))

    if set(correct_act)-priority_act!=set() or priority_act-set(correct_act)!=set():
        break



from btgym.envs.virtualhometext.exec_lib._base.VHTAction_small import VHTAction_small
from btgym.envs.virtualhome.exec_lib._base.VHAction import VHAction

from btgym.utils.tools import collect_action_nodes
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from btgym.algos.llm_client.tools import goal_transfer_str, act_format_records
from tools import *


# VirtualHome
from btgym.envs.virtualhome.exec_lib._base.VHAction import VHAction
env = btgym.make("VH-PutMilkInFridge")
cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
cur_cond_set |= {f'IsClose({arg})' for arg in VHAction.CanOpenPlaces}
cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHAction.HasSwitchObjects}
print(f"共收集到 {len(VHAction.AllObject)} 个物体")

# RobotHowSmall
# from btgym.envs.virtualhometextsmall.exec_lib._base.VHTAction import VHTAction as RHS
# env = btgym.make("VHT-Small")
# cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
# cur_cond_set |= {f'IsClose({arg})' for arg in RHS.CAN_OPEN}
# cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in RHS.HAS_SWITCH}
# cur_cond_set |= {f'IsUnplugged({arg})' for arg in RHS.HAS_PLUG}
# print(f"共收集到 {len(RHS.AllObject)} 个物体")


# RobotHowBig
# from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction as RHB
# env = btgym.make("VHT-PutMilkInFridge")
# cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
# cur_cond_set |= {f'IsClose({arg})' for arg in RHB.CAN_OPEN}
# cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in RHB.HAS_SWITCH}
# cur_cond_set |= {f'IsUnplugged({arg})' for arg in RHB.HAS_PLUG}
# print(f"共收集到 {len(RHB.AllObject)} 个物体")


# goal=['IsOn_apple_desk']
goal=['IsIn_apple_fridge']
# priority_act_ls=["Walk_milk"]
goal_set = goal_transfer_str(' & '.join(goal))

algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                      priority_act_ls=[], key_predicates=[],
                      key_objects=[],
                      selected_algorithm="opt", mode="big",
                      llm_reflect=False, time_limit=30,
                      heuristic_choice=0)
expanded_num, planning_time_total, cost, error, act_num, current_cost, record_act_ls = \
    execute_algorithm(algo, goal_set, cur_cond_set)
time_limit_exceeded = algo.algo.time_limit_exceeded

success = not error and not time_limit_exceeded

_priority_act_ls, key_predicates, key_objects = act_format_records(record_act_ls)
priority_act_ls = record_act_ls
# 打印所有变量
# 定义蓝色的ANSI转义序列
BLUE = "\033[94m"
RESET = "\033[0m"

print(f"{BLUE}Try to use big space...{RESET}")

# 打印指定的三个输出，并使用蓝色
print(f"{BLUE}success:{RESET}", success)
print(f"{BLUE}goal:{RESET}", ' & '.join(goal))
print(f"{BLUE}_priority_act_ls:{RESET}", _priority_act_ls)
print(f"{BLUE}act_num:{RESET}", act_num)
print(f"{BLUE}planning_time_total:{RESET}", planning_time_total)
print(f"{BLUE}expanded_num:{RESET}", expanded_num)
print(f"{BLUE}current_cost:{RESET}", current_cost)
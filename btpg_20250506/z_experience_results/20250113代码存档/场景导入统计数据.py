from btgym.envs.RoboWaiter.exec_lib._base.RWAction import RWAction
import btgym
from btgym.utils.tools import collect_action_nodes



# ===================== RoboWaiter ========================
from btgym.envs.RoboWaiter.exec_lib._base.RWAction import RWAction
env = btgym.make("RWEnv")
cur_cond_set = env.agents[0].condition_set = {'RobotNear(Bar)', 'Holding(Nothing)'}
cur_cond_set |= {f'Exists({arg})' for arg in RWAction.all_object - {'Coffee', 'Water', 'Dessert'}}
cur_cond_set |= {f'Exists({arg})' for arg in RWAction.all_object - {'Coffee', 'Water', 'Dessert'}}
big_actions = collect_action_nodes(env.behavior_lib)
print(f"共收集到 {len(RWAction.AllObject)} 个物体")

# ===================== VirtualHome ========================
# from btgym.envs.VirtualHome.exec_lib._base.VHAction import VHAction
#
# env = btgym.make("VH-PutMilkInFridge")
# cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)",
#                                               "IsStanding(self)"}
# cur_cond_set |= {f'IsClose({arg})' for arg in VHAction.CanOpenPlaces}
# cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHAction.HasSwitchObjects}
# big_actions = collect_action_nodes(env.behavior_lib)
# print(f"共收集到 {len(VHAction.AllObject)} 个物体")


# ===================== RobotHow-Small ========================
# from btgym.envs.RobotHow_Small.exec_lib._base.OGAction import OGAction
# env = btgym.make("VHT-Small")
# cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)",
#                                               "IsStanding(self)"}
# cur_cond_set |= {f'IsClose({arg})' for arg in OGAction.CAN_OPEN}
# cur_cond_set |= {f'IsUnplugged({arg})' for arg in OGAction.HAS_PLUG}
# cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in OGAction.HAS_SWITCH}
# big_actions = collect_action_nodes(env.behavior_lib)


# ===================== RobotHow ========================
from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction as RHB

env = btgym.make("VHT-PutMilkInFridge")
cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)",
                                              "IsStanding(self)"}
cur_cond_set |= {f'IsClose({arg})' for arg in RHB.CAN_OPEN}
cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in RHB.HAS_SWITCH}
cur_cond_set |= {f'IsUnplugged({arg})' for arg in RHB.HAS_PLUG}
print(f"Collected a total of {len(RHB.AllObject)} objects")
big_actions = collect_action_nodes(env.behavior_lib)


# 计算有限状态数
conds_set=set()
for act in big_actions:
    conds_set |= act.pre
    conds_set |= act.del_set
    conds_set |= act.add
print("len(conds_set)",len(conds_set))

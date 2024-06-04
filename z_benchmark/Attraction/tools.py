import os
from btgym.utils import ROOT_PATH
os.chdir(f'{ROOT_PATH}/../')
import btgym
from btgym.utils.tools import collect_action_nodes
import copy
import random

from btgym.envs.RoboWaiter.exec_lib._base.RWAction import RWAction as RW
from btgym.envs.virtualhome.exec_lib._base.VHAction import VHAction as VH
from btgym.envs.RobotHow_Small.exec_lib._base.RHSAction import RHSAction as RHS
from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction as RH


def modify_condition_set(sence,SimAct, cur_cond_set):
    new_cur_state = copy.deepcopy(cur_cond_set)

    # 改变位置
    near_conditions = [cond for cond in new_cur_state if cond.startswith('IsNear')]
    if near_conditions:
        # 已有位置，随机决定是改变还是移除
        if random.choice([True, False]):
            new_cur_state.remove(random.choice(near_conditions))
        new_position = random.choice(list(SimAct.AllObject))
        new_cur_state.add(f'IsNear(self,{new_position})')
    else:
        # 没有位置，添加一个
        new_position = random.choice(list(SimAct.AllObject))
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
                    put_position = random.choice(list(SimAct.SurfacePlaces))
                    new_cur_state.add(f'IsOn({item},{put_position})')
                else:
                    put_position = random.choice(list(SimAct.CanPutInPlaces))
                    new_cur_state.add(f'IsIn({item},{put_position})')

            else:
                # 换一个物体
                new_cur_state.discard(holding_condition)
                new_object = random.choice(list(SimAct.GRABBABLE))
                new_cur_state.add(f'Is{hand}Holding(self,{new_object})')
                new_cur_state.discard(f'Is{hand}HandEmpty(self)')
        else:
            if random.choice([True, False]):
                new_object = random.choice(list(SimAct.GRABBABLE))
                new_cur_state.add(f'Is{hand}Holding(self,{new_object})')
                new_cur_state.discard(f'Is{hand}HandEmpty(self)')


    # 随机改变 物体的固有属性
    # 处理可开关的物体（如门、窗、设备等）
    for obj in SimAct.HAS_SWITCH:
        state = 'IsSwitchedOn' if random.choice([True, False]) else 'IsSwitchedOff'
        opposite_state = 'IsSwitchedOff' if state == 'IsSwitchedOn' else 'IsSwitchedOn'
        # 添加新状态前删除相反的状态
        new_cur_state.discard(f'{opposite_state}({obj})')
        new_cur_state.add(f'{state}({obj})')

    # 处理可开关或靠近的物体（如柜子、抽屉等）
    for obj in SimAct.CAN_OPEN:
        state = 'IsOpen' if random.choice([True, False]) else 'IsClose'
        opposite_state = 'IsClose' if state == 'IsOpen' else 'IsOpen'
        # 添加新状态前删除相反的状态
        new_cur_state.discard(f'{opposite_state}({obj})')
        new_cur_state.add(f'{state}({obj})')

    if sence in ["RHS","RH"]:
        for obj in SimAct.HAS_PLUG:
            state = 'IsPlugged' if random.choice([True, False]) else 'IsUnplugged'
            opposite_state = 'IsUnplugged' if state == 'IsPlugged' else 'IsPlugged'
            # 添加新状态前删除相反的状态
            new_cur_state.discard(f'{opposite_state}({obj})')
            new_cur_state.add(f'{state}({obj})')


    return new_cur_state

def setup_environment(scene):
    if scene == "RW":
        # ===================== RoboWaiter ========================
        from btgym.envs.RoboWaiter.exec_lib._base.RWAction import RWAction
        env = btgym.make("RWEnv")
        cur_cond_set = env.agents[0].condition_set = {'RobotNear(Bar)', 'Holding(Nothing)'}
        cur_cond_set |= {f'Exists({arg})' for arg in RWAction.all_object - {'Coffee', 'Water', 'Dessert'}}
        cur_cond_set |= {f'Exists({arg})' for arg in RWAction.all_object - {'Coffee', 'Water', 'Dessert'}}
        big_actions = collect_action_nodes(env.behavior_lib)
        return env,cur_cond_set

    elif scene == "VH":
        # ===================== VirtualHome ========================
        from btgym.envs.virtualhome.exec_lib._base.VHAction import VHAction
        env = btgym.make("VH-PutMilkInFridge")
        cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)",
                                                      "IsStanding(self)"}
        cur_cond_set |= {f'IsClose({arg})' for arg in VHAction.CanOpenPlaces}
        cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHAction.HasSwitchObjects}
        big_actions = collect_action_nodes(env.behavior_lib)
        return env,cur_cond_set

    elif scene == "RHS":
        # ===================== RobotHow-Small ========================
        from btgym.envs.RobotHow_Small.exec_lib._base.RHSAction import RHSAction
        env = btgym.make("VHT-Small")
        cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)",
                                                      "IsStanding(self)"}
        cur_cond_set |= {f'IsClose({arg})' for arg in RHSAction.CAN_OPEN}
        cur_cond_set |= {f'IsUnplugged({arg})' for arg in RHSAction.HAS_PLUG}
        cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in RHSAction.HAS_SWITCH}
        big_actions = collect_action_nodes(env.behavior_lib)
        return env,cur_cond_set

    elif scene == "RH":
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
        return env,cur_cond_set
    else:
        print(f"\033[91mCannot parse scene: {scene}\033[0m")
        return None,None



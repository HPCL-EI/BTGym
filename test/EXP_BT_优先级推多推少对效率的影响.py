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

env = btgym.make("VHT-Small")

action_list=[]
for cls in env.behavior_lib["Action"].values():
    if cls.can_be_expanded:
        print(f"可扩展动作：{cls.__name__}, 存在{len(cls.valid_args_small)}个有效论域组合")
        if cls.num_args == 0:
            action_list.append(Action(name=cls.get_ins_name(), **cls.get_info()))
        if cls.num_args == 1:
            for arg in cls.valid_args_small:
                action_list.append(Action(name=cls.get_ins_name(arg), **cls.get_info(arg)))
        if cls.num_args > 1:
            for args in cls.valid_args_small:
                action_list.append(Action(name=cls.get_ins_name(*args), **cls.get_info(*args)))
print(f"共收集到{len(action_list)}个实例化动作:")
all_actions_set = set()
for act in action_list:
    all_actions_set.add(act.name)

goal_set = [{'IsIn(milk,microwave)','IsSwitchedOn(microwave)'}]
true_priority_act_set = {"Walk(milk)", "RightGrab(milk)", "Walk(microwave)", "Open(microwave)", \
                    "RightPutIn(milk,microwave)",'SwitchOn(microwave)'}

error_priority_act_set = all_actions_set-true_priority_act_set
# 推荐优先级

priority_act_ls=set()
error_rate = 0.5
correct_rate = 0.5

error_num=int(len(true_priority_act_set)*error_rate)
correct_num=int(len(true_priority_act_set)*correct_rate)

priority_act_ls |= set(random.sample(error_priority_act_set, error_num))
priority_act_ls |= set(random.sample(true_priority_act_set, correct_num))

print(priority_act_ls)


cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"
                                              }
cur_cond_set |= {f'IsClose({arg})' for arg in VHTAction_small.CAN_OPEN}
cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHTAction_small.HAS_SWITCH}
cur_cond_set |= {f'IsUnplugged({arg})' for arg in VHTAction_small.HAS_PLUG}
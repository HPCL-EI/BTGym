import time

from btgym import BehaviorTree
from btgym import ExecBehaviorLibrary
import btgym
from btgym.utils import ROOT_PATH
from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
from btgym.algos.bt_autogen.main_interface import BTExpInterface

from sympy import symbols, Not, Or, And, to_dnf
from sympy import symbols, simplify_logic
import re

from btgym.algos.llm_client.tools import goal_transfer_str,act_str_process
from btgym.envs.VirtualHome.exec_lib._base.VHAction import  VHAction


env = btgym.make("VH-PutMilkInFridge")


#todo: LLMs
prompt_file = f"{ROOT_PATH}\\algos\\llm_client\\prompt.txt"
with open(prompt_file, 'r', encoding="utf-8") as f:
    prompt = f.read().strip()
# print(prompt)

# instuction = "Put the bowl in the dishwasher and wash it."
# instuction="Put the milk and chicken in the fridge."
instuction="Turn on the computer, TV, and lights, then put the bowl in the dishwasher and wash it"

# llm = LLMGPT3()
# question = prompt+instuction
# answer = llm.request(question=question)
# print(answer)
#
# goal_str = answer.split("Actions:")[0].replace("Goals:", "").strip()
# act_str = answer.split("Actions:")[1]

# goal_set = goal_transfer_str(goal_str)
# priority_act_ls = act_str_process(act_str)

# print("goal",goal_set)
# print("act:",priority_act_ls)


# goal_set=[{"IsOn(towel,towelrack)"}]
# goal_set=[{"IsSittingOn(self,Toilet)"}]
# goal_set=[{"IsSwitchedOn(Cellphone)"}]

goal_set=[{"IsOpen(Nightstand)"}]

priority_act_ls=[]

#todo: BTExp:process
cur_cond_set=env.agents[0].condition_set = {"IsSwitchedOff(tv)","IsSwitchedOff(faucet)","IsSwitchedOff(stove)", "IsSwitchedOff(dishwasher)",
                                            "IsSwitchedOn(lightswitch)","IsSwitchedOn(tablelamp)",
                                            "IsSwitchedOff(coffeemaker)","IsSwitchedOff(toaster)","IsSwitchedOff(microwave)",
                                            "IsSwitchedOff(computer)","IsSwitchedOff(radio)","IsSwitchedOff(wallphone)",

                                            "IsClose(fridge)","IsClose(bathroomcabinet)","IsClose(stove)","IsClose(dishwasher)","IsClose(microwave)",
                                            "IsOpen(bookshelf)","IsClose(cabinet)","IsClose(bathroomcabinet)"
                                            ,"IsClose(door)",
                                            "IsClose(toilet)",

                               "IsRightHandEmpty(self)","IsLeftHandEmpty(self)","IsStanding(self)"
                               }
cur_cond_set |= {f"IsClose({obj})" for obj in VHAction.CanOpenPlaces}

start_time = time.time()
algo = BTExpInterface(env.behavior_lib, cur_cond_set,priority_act_ls)
ptml_string = algo.process(goal_set)
end_time = time.time()
planning_time_total = (end_time-start_time)
print("planning_time_total:",planning_time_total)

file_name = "grasp_milk"
file_path = f'./{file_name}.btml'
with open(file_path, 'w') as file:
    file.write(ptml_string)


# algo2 = BTExpInterface(env.behavior_lib, cur_cond_set,bt_algo_opt=False)
# ptml_string2 = algo2.process(goal)
# file_name2 = "grasp_milk_baseline"
# file_path2 = f'./{file_name2}.btml'
# with open(file_path2, 'w') as file:
#     file.write(ptml_string2)


# 读取执行
bt = BehaviorTree(file_name+".btml", env.behavior_lib)
bt.print()
bt.draw()

env.agents[0].bind_bt(bt)
env.reset()
env.print_ticks = False

is_finished = False
while not is_finished:
    is_finished = env.step()
    # print(env.agents[0].condition_set)

    g_finished=True
    for g in goal_set:
        if not g<= env.agents[0].condition_set:
            g_finished=False
        if g_finished:
            is_finished=True
env.close()
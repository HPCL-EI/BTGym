import time

from btgym import BehaviorTree
from btgym import ExecBehaviorLibrary
import btgym

from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
from btgym.algos.bt_autogen.main_interface import BTExpInterface


env = btgym.make("VH-PutMilkInFridge")


#todo: LLMs

# question="请把厨房桌上的牛奶放到冰箱里,并关上冰箱。"
question="请打开电视"
llm = LLMGPT3()
prompt=""
# goal = llm.request(question=prompt+question)
# goal=[{"IsWatching(self,tv)"}]
# goal=[{"IsIn(milk,fridge)","IsClosed(fridge)"}]  # goal 是 set组成的列表

# goal=[{"IsIn(milk,fridge)"}]
goal=[{"IsOpened(fridge)"}]
print("goal",goal)

#todo: BTExp
#todo: BTExp:LoadActions
# action_list=None

#todo: BTExp:process
cur_cond_set=env.agents[0].condition_set = {"IsSwitchedOff(tv)","IsClosed(fridge)",
                               "IsRightHandEmpty(self)","IsLeftHandEmpty(self)","IsStanding(self)"
                               }

algo = BTExpInterface(env.behavior_lib, cur_cond_set)
ptml_string = algo.process(goal)

file_name = "grasp_milk"
file_path = f'./{file_name}.btml'
with open(file_path, 'w') as file:
    file.write(ptml_string)



# 读取执行
bt = BehaviorTree("grasp_milk.btml", env.behavior_lib)
bt.print()

env.agents[0].bind_bt(bt)
env.reset()

is_finished = False
while not is_finished:
    is_finished = env.step()
    # print(env.agents[0].condition_set)

    g_finished=True
    for g in goal:
        if not g<= env.agents[0].condition_set:
            g_finished=False
    if g_finished:
        is_finished=True
env.close()
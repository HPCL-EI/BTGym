import time

from btgym import BehaviorTree
from btgym import ExecBehaviorLibrary
import btgym

env = btgym.make("VH-PutMilkInFridge")
bt = BehaviorTree("put_milk_in_fridge.btml", env.behavior_lib)

bt.print()

env.agents[0].bind_bt(bt)
env.reset()

env.agents[0].condition_set = {"IsClosed(fridge)",
                               "IsRightHandEmpty(self)","IsLeftHandEmpty(self)","IsStanding(self)",
                               }

is_finished = False
while not is_finished:
    is_finished = env.step()
    print(env.agents[0].condition_set)

env.close()
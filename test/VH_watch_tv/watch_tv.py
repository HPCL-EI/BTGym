import time

from btgym import BehaviorTree
from btgym import ExecBehaviorLibrary
import btgym


# behavior_tree = BehaviorTree("Default.btml")
# behavior_tree.print()
# behavior_tree.draw()

# lib_path = f'{btgym.ROOT_PATH}/exec_lib'
# exec_lib = ExecBehaviorLibrary(lib_path)
# print(exec_lib.Action)

# exec_bt = ExecBehaviorTree("Default.btml",exec_lib)


env = btgym.make("VH-WatchTV")
bt = BehaviorTree("WatchTV.btml", env.behavior_lib)
bt.print()

env.agents[0].bind_bt(bt)
env.reset()

env.agents[0].condition_set = set()

is_finished = False
while not is_finished:
    is_finished = env.step()
    print(env.agents[0].condition_set)

env.close()
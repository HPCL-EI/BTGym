import time

from btgym import BehaviorTree
from btgym import ExecBehaviorLibrary
import btgym



env = btgym.make("VHT-WatchTV")
bt = BehaviorTree("VH_watch_tv/WatchTV.btml", env.behavior_lib)
bt.print()
print()

env.agents[0].bind_bt(bt)
env.reset()

is_finished = False
while not is_finished:
    is_finished = env.step()
    # print(env.agents[0].condition_set)

env.close()

print("\n====== batch scripts ======\n")


env = btgym.make("VHT-WatchTV")
# script = ['<char0> [Find] <tv> (1)',
#           '<char0> [switchon] <tv> (1)',
#           '<char0> [Walk] <sofa> (1)',
#           '<char0> [Sit] <sofa> (1)',
#           '<char0> [Watch] <tv> (1)']
script = ['<char0> [Find] <waterglass> (1)',
          '<char0> [Pour] <waterglass> (1) <coffeepot> (1)']
env.run_script(script)
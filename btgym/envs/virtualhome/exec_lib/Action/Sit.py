import py_trees as ptree

from typing import Any
from btgym.envs.virtualhome.exec_lib._base.VHAction import VHAction
from btgym.behavior_tree import Status

class Sit(VHAction):
    can_be_expanded = True
    num_args = 1

    def change_condition_set(self):
        self.agent.condition_set.add(f"IsSittingOn(self,{self.args[0]})")
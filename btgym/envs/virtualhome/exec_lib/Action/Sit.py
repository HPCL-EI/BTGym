import py_trees as ptree

from typing import Any
from btgym.envs.virtualhome.exec_lib._base.VHAction import VHAction
from btgym.behavior_tree import Status

class Sit(VHAction):
    can_be_expanded = True
    num_args = 1
    valid_args=set()

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"]={"IsStanding(self)",f"IsNear({arg[0]})"}
        info["add"]={f"IsSittingOn(self,{arg[0]})",f"IsSitting(self)"}
        info["del_set"] = {f"IsStanding(self)"}
        return info

    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]
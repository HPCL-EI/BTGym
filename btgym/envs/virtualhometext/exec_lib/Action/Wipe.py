from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction

class Wipe(VHTAction):
    can_be_expanded = True
    num_args = 1
    valid_args = VHTAction.AllObject - VHTAction.EATABLE - VHTAction.DRINKABLE

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"]={f"IsHoldingCleaningTool(self)",f"IsNear(self,{arg[0]})"} # IsLeftHandEmpty()至少有一只手是空闲的
        info["add"]={f"IsClean({arg[0]})"}
        info["del_set"] = set()
        info["cost"] = 9
        return info


    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]

        # self.agent.condition_set.add(f"IsSwitchedOn({self.args[0]})")
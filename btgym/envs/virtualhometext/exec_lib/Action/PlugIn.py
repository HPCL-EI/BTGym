from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction
class PlugIn(VHTAction):
    can_be_expanded = True
    num_args = 1
    valid_args = VHTAction.HAS_PLUG


    def __init__(self, *args):
        super().__init__(*args)
        self.target_obj = self.args[0]

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"]={"IsLeftHandEmpty(self)",f"IsNear(self,{arg[0]})",f"IsUnplugged({arg[0]})"} # IsLeftHandEmpty()至少有一只手是空闲的
        info["add"]={f"IsPlugged({arg[0]})"}
        info["del_set"] = {f"IsUnplugged({arg[0]})"}
        info["cost"] = 8
        return info


    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]

        # self.agent.condition_set.add(f"IsSwitchedOn({self.args[0]})")
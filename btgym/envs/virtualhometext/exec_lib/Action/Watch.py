from btgym.envs.virtualhome.exec_lib._base.VHAction import VHAction

class Watch(VHAction):
    can_be_expanded = True
    num_args = 1

    def change_condition_set(self):
        self.agent.condition_set.add(f"IsWatching(self,{self.args[0]})")
from btgym.envs.virtualhome.exec_lib._base.VHAction import VHAction

class Walk(VHAction):
    can_be_expanded = True
    num_args = 1

    def change_condition_set(self):
        del_list = []
        for c in self.agent.condition_set:
            if "IsNear" in c:
                del_list.append(c)
        for c in del_list:
            self.agent.condition_set.remove(c)

        self.agent.condition_set.add(f"IsNear(self,{self.args[0]})")
from btgym.envs.robowaiter.exec_lib._base.VHTAction import VHTAction

class PickUp(VHTAction):
    can_be_expanded = True
    num_args = 1
    valid_args = VHTAction.all_object
    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"] = {f'RobotNear({arg})','Holding(Nothing)'}
        info["add"] = {f'Holding({arg})'}
        info["del_set"] = {f'Holding(Nothing)',f'Exists({arg})'}
        for place in cls.all_place:
            info["del_set"] |= {f'On({arg},{place})'}
        info['cost'] = 2
        return info

    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]

        # self.agent.condition_set.add(f"IsSwitchedOn({self.args[0]})")
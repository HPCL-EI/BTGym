from btgym.behavior_tree.base_nodes import Action
from btgym.behavior_tree import Status

class VHAction(Action):
    can_be_expanded = True
    num_args = 1

    def change_condition_set(self):
        pass

    def update(self) -> Status:
        script = [f'<char0> [{self.__class__.__name__.lower()}] <{self.args[0].lower()}> (1)']

        self.env.run_script(script)
        self.change_condition_set()

        return Status.RUNNING
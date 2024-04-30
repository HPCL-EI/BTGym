

from btgym.envs.virtualhome.exec_lib import Action
from btgym.behavior_tree import Status

class Sit(Action.VHSit):
    can_be_expanded = True
    num_args = 1


    def update(self):
        pass
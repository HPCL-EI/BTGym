from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction
import itertools

class Put(VHTAction):
    can_be_expanded = False
    num_args = 2
    valid_args = list(itertools.product(VHTAction.Objects, VHTAction.SurfacePlaces))

    def __init__(self, *args):
        super().__init__(*args)

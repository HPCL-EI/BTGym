from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction

class Grab(VHTAction):
    can_be_expanded = False
    num_args = 1
    valid_args = VHTAction.Objects

    def __init__(self, *args):
        super().__init__(*args)

from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction
from btgym.envs.virtualhometext.exec_lib._base.VHTAction_small import VHTAction_small

import itertools

class Put(VHTAction):
    can_be_expanded = False
    num_args = 2
    # valid_args = list(itertools.product(VHTAction.GRABBABLE, VHTAction.SURFACES))

    set_1_food = VHTAction.GRABBABLE & (VHTAction.EATABLE|VHTAction.DRINKABLE|{"bananas",'chicken','cutlets','breadslice','chips','chocolatesyrup',
                 'milk','wine',"cereal"})

    valid_args = list(itertools.product(VHTAction.GRABBABLE-set_1_food, VHTAction.SURFACES-{"towelrack","plate","fryingpan"})) \
                    + list(itertools.product(VHTAction.GRABBABLE & {'towel'}, {"towelrack"})) \
                    + list(itertools.product(set_1_food, VHTAction.SURFACES-{"towelrack","bathroomcounter"}))

    valid_args_small = list(itertools.product(VHTAction_small.GRABBABLE, VHTAction_small.SURFACES))

    def __init__(self, *args):
        super().__init__(*args)

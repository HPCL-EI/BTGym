from btgym.behavior_tree.base_nodes import Action
from btgym.behavior_tree import Status

class VHAction(Action):
    can_be_expanded = True
    num_args = 1
    # SurfacePlaces = {"fridge","desk","kitchentable","coffeetable","kitchencabinet","kitchencounter","oventray"}
    # SittablePlaces = {"fridge", "desk", "kitchentable", "coffeetable", "kitchencabinet", "kitchencounter","oventray"}
    # CanOpenPlaces= {"fridge", "kitchencabinet"}
    # Objects={"milk","apple","cereal","mug","tv"}
    # HasSwitchObjects = {"tv"}

    SurfacePlaces = set()
    SittablePlaces = set()
    CanOpenPlaces= {"fridge"}
    CanPutInPlaces={"fridge"}
    Objects={"milk"}
    HasSwitchObjects = set()

    @property
    def action_class_name(self):
        return self.__class__.__name__


    def change_condition_set(self):
        pass

    def update(self) -> Status:
        # script = [f'<char0> [{self.__class__.__name__.lower()}] <{self.args[0].lower()}> (1)']

        if self.num_args==1:
            script = [f'<char0> [{self.action_class_name.lower()}] <{self.args[0].lower()}> (1)']
        else:
            script = [f'<char0> [{self.action_class_name.lower()}] <{self.args[0].lower()}> (1) <{self.args[1].lower()}> (1)']


        self.env.run_script(script)
        print("script: ",script)
        self.change_condition_set()

        return Status.RUNNING
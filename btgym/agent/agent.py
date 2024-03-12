

class Agent(object):
    def __init__(self):
        self.condition_set = set()

    def bind_bt(self,bt):
        self.bt = bt
        bt.bind_agent(self)

from btgym.algos.bt_autogen.Action import  Action,state_transition
from btgym.algos.bt_autogen.OptimalBTExpansionAlgorithm import OptBTExpAlgorithm
from btgym.algos.bt_autogen.BTExpansionAlgorithm import BTalgorithm
from btgym.algos.bt_autogen.BTExpansionAlgorithmBFS import BTalgorithmBFS
from btgym.algos.bt_autogen.BTExpansionAlgorithmDFS import BTalgorithmDFS
from btgym.algos.bt_autogen.OptimalBTExpansionAlgorithmHeuristics import OptBTExpAlgorithmHeuristics
from btgym.algos.bt_autogen.examples import *


# 封装好的主接口
class BTExpInterface:
    def __init__(self, behavior_lib,cur_cond_set,priority_act_ls=[],selected_algorithm="opt",bt_algo_opt=True):
        """
        Initialize the BTOptExpansion with a list of actions.
        :param action_list: A list of actions to be used in the behavior tree.
        """
        self.cur_cond_set = cur_cond_set
        self.bt_algo_opt = bt_algo_opt
        self.selected_algorithm = selected_algorithm


        self.actions = collect_action_nodes(behavior_lib)

        self.priority_act_ls = priority_act_ls
        self.actions = self.adjust_action_priority(self.actions,self.priority_act_ls,self.selected_algorithm)

        self.has_processed = False

        self.min_cost=float("inf")


    def process(self, goal):
        """
        Process the input sets and return a string result.
        :param input_set: The set of goal states and the set of initial states.
        :return: A btml string representing the outcome of the behavior tree.
        """
        self.goal = goal
        # if self.bt_algo_opt:
        #     self.algo = OptBTExpAlgorithm(verbose=False)
        # else:
        #     self.algo = BTalgorithm(verbose=False)
        if self.selected_algorithm=="opt":
            self.algo = OptBTExpAlgorithm(verbose=False)

        elif self.selected_algorithm=="opt-h":
            self.algo = OptBTExpAlgorithmHeuristics(verbose=False)
        elif self.selected_algorithm=="bfs":
            self.algo = BTalgorithmBFS(verbose=False)
            # self.algo = BTalgorithm(verbose=False)
        elif self.selected_algorithm=="dfs":
            self.algo = BTalgorithmDFS(verbose=False)
        else:
            print("Error in algorithm selection: This algorithm does not exist.")

        self.algo.clear()
        self.algo.run_algorithm(self.cur_cond_set,self.goal, self.actions) # 调用算法得到行为树保存至 algo.bt


        # self.btml_string = self.algo.get_btml()
        self.has_processed = True
        # algo.print_solution() # print behavior tree

        # return self.btml_string
        return True

    def post_process(self):
        self.btml_string = self.algo.get_btml()
        if self.selected_algorithm == "opt":
            self.min_cost = self.algo.min_cost
        else:
            self.min_cost = self.algo.get_cost()
        return self.btml_string,self.min_cost


    # 方法一：查找所有初始状态是否包含当前状态
    def find_all_leaf_states_contain_start(self,start):
        if not self.has_processed:
            raise RuntimeError("The process method must be called before find_all_leaf_states_contain_start!")
        # 返回所有能到达目标状态的初始状态
        state_leafs = self.algo.get_all_state_leafs()
        for state in state_leafs:
            if start >= state:
                return True
        return False

    # 方法二：模拟跑一遍行为树，看 start 能够通过执行一系列动作到达 goal
    def run_bt_from_start(self,goal,start):
        if not self.has_processed:
            raise RuntimeError("The process method must be called before run_bt_from_start!")
        # 检查是否能到达目标
        right_bt = True
        state = start
        steps = 0
        val, obj = self.algo.bt.tick(state)
        while val != 'success' and val != 'failure':
            state = state_transition(state, obj)
            val, obj = self.algo.bt.tick(state)
            if (val == 'failure'):
                # print("bt fails at step", steps)
                right_bt = False
            steps += 1
        if not goal <= state:
            # print("wrong solution", steps)
            right_bt = False
        else:
            pass
            # print("right solution", steps)
        return right_bt


    def adjust_action_priority(self,action_list,priority_act_ls,selected_algorithm):
        # recommended_acts=["RightPutIn(bananas,fridge)",
        #                   "Open(fridge)",
        #                   "Walk(fridge)",
        #                   "Close(fridge)",
        #                   "RightGrab(bananas)",
        #                   "Walk(bananas)"
        #                   ]

        recommended_acts = priority_act_ls

        if selected_algorithm == "opt-h":
            for act in action_list:
                if act.name in recommended_acts:
                    act.priority = 0
        else:
            for act in action_list:
                if act.name in recommended_acts:
                    act.cost=0
                    act.priority=0

        # 对action排序
        action_list.sort(key=lambda x: x.cost)

        return action_list


def collect_action_nodes(behavior_lib):
    action_list = []

    for cls in behavior_lib["Action"].values():
        if cls.can_be_expanded:
            print(f"可扩展动作：{cls.__name__}, 存在{len(cls.valid_args)}个有效论域组合")
            if cls.num_args == 0:
                action_list.append(Action(name=cls.get_ins_name(), **cls.get_info()))
            if cls.num_args == 1:
                for arg in cls.valid_args:
                    action_list.append(Action(name=cls.get_ins_name(arg), **cls.get_info(arg)))
            if cls.num_args > 1:
                for args in cls.valid_args:
                    action_list.append(Action(name=cls.get_ins_name(*args), **cls.get_info(*args)))

    print(f"共收集到{len(action_list)}个实例化动作:")
    # for a in self.action_list:
    #     if "Turn" in a.name:
    #         print(a.name)
    print("--------------------\n")

    return action_list







if __name__ == '__main__' :

    # todo: Example Cafe
    # todo: Define goal, start, actions
    actions=[
        Action(name='PutDown(Table,Coffee)', pre={'Holding(Coffee)','At(Robot,Table)'}, add={'At(Table,Coffee)','NotHolding'}, del_set={'Holding(Coffee)'}, cost=1),
        Action(name='PutDown(Table,VacuumCup)', pre={'Holding(VacuumCup)','At(Robot,Table)'}, add={'At(Table,VacuumCup)','NotHolding'}, del_set={'Holding(VacuumCup)'}, cost=1),

        Action(name='PickUp(Coffee)', pre={'NotHolding','At(Robot,Coffee)'}, add={'Holding(Coffee)'}, del_set={'NotHolding'}, cost=1),

        Action(name='MoveTo(Table)', pre={'Available(Table)'}, add={'At(Robot,Table)'}, del_set={'At(Robot,FrontDesk)','At(Robot,Coffee)','At(Robot,CoffeeMachine)'}, cost=1),
        Action(name='MoveTo(Coffee)', pre={'Available(Coffee)'}, add={'At(Robot,Coffee)'}, del_set={'At(Robot,FrontDesk)','At(Robot,Table)','At(Robot,CoffeeMachine)'}, cost=1),
        Action(name='MoveTo(CoffeeMachine)', pre={'Available(CoffeeMachine)'}, add={'At(Robot,CoffeeMachine)'}, del_set={'At(Robot,FrontDesk)','At(Robot,Coffee)','At(Robot,Table)'}, cost=1),

        Action(name='OpCoffeeMachine', pre={'At(Robot,CoffeeMachine)','NotHolding'}, add={'Available(Coffee)','At(Robot,Coffee)'}, del_set=set(), cost=1),
    ]
    algo = BTOptExpInterface(actions)


    goal = {'At(Table,Coffee)'}
    btml_string = algo.process(goal)
    print(btml_string)

    file_name = "sub_task"
    with open(f'./{file_name}.btml', 'w') as file:
        file.write(btml_string)


    # 判断初始状态能否到达目标状态
    start = {'At(Robot,Bar)', 'Holding(VacuumCup)', 'Available(Table)', 'Available(CoffeeMachine)','Available(FrontDesk)'}
    # 方法一：算法返回所有可能的初始状态，在里面看看有没有对应的初始状态
    right_bt = algo.find_all_leaf_states_contain_start(start)
    if not right_bt:
        print("ERROR1: The current state cannot reach the goal state!")
    else:
        print("Right1: The current state can reach the goal state!")
    # 方法二：预先跑一边行为树，看能否到达目标状态
    right_bt2 = algo.run_bt_from_start(goal,start)
    if not right_bt2:
        print("ERROR2: The current state cannot reach the goal state!")
    else:
        print("Right2: The current state can reach the goal state!")

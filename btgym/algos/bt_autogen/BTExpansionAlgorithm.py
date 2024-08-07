import random
import numpy as np
import copy
import time
from btgym.algos.bt_autogen.behaviour_tree import Leaf,ControlBT
from btgym.algos.bt_autogen.Action import Action,generate_random_state,state_transition




# 本文所提出的完备规划算法
class BTalgorithm:
    def __init__(self,verbose=False):
        self.bt = None
        self.nodes = []
        self.traversed = []
        self.conditions = []
        self.conditions_index = []
        self.verbose = verbose
        # print (self.conditions_list[0])

    def clear(self):
        self.bt = None
        self.nodes = []
        self.traversed = []
        self.conditions = []
        self.conditions_index = []

    # 运行规划算法，从初始状态、目标状态和可用行动，计算行为树self.bt
    def run_algorithm_selTree(self, start, goal, actions):
        # 初始行为树只包含目标条件
        bt = ControlBT(type='cond')
        g_node = Leaf(type='cond', content=goal,min_cost=0)
        bt.add_child([g_node])

        self.conditions.append(goal)
        self.nodes.append(g_node)  # condition node list
        # 尝试在初始状态执行行为树
        val, obj = bt.tick(start)
        canrun = False
        if val == 'success' or val == 'running':
            canrun = True
        # 循环扩展，直到行为树能够在初始状态运行
        while not canrun:
            index = -1
            for i in range(0, len(self.nodes)):
                if self.nodes[i].content in self.traversed:
                    continue
                else:
                    c_node = self.nodes[i]
                    index = i
                    break
            if index == -1:  # 树中结点扩展完毕，仍无法运行行为树，返回失败
                print('Failure')
                return False
            # 根据所选择条件结点扩展子树
            subtree = ControlBT(type='?')
            subtree.add_child([copy.deepcopy(c_node)])  # 子树首先保留所扩展结点
            c = c_node.content  # 子树所扩展结点对应的条件（一个文字的set）

            for i in range(0, len(actions)):  # 选择符合条件的行动，
                # print("have action")
                if not c & ((actions[i].pre | actions[i].add) - actions[i].del_set) <= set():
                    # print ("pass add")
                    if (c - actions[i].del_set) == c:
                        # print("pass delete")
                        c_attr = (actions[i].pre | c) - actions[i].add
                        valid = True

                        # 这样剪枝存在错误性
                        # if conflict(c_attr):
                        #     continue

                        for j in self.traversed:  # 剪枝操作
                            if j <= c_attr:
                                valid = False
                                break

                        if valid:
                            # print("pass prune")
                            # 构建行动的顺序结构
                            sequence_structure = ControlBT(type='>')
                            c_attr_node = Leaf(type='cond', content=c_attr, min_cost=0)
                            a_node = Leaf(type='act', content=actions[i], min_cost=0)
                            sequence_structure.add_child([c_attr_node, a_node])
                            # 将顺序结构添加到子树
                            subtree.add_child([sequence_structure])

                            self.nodes.append(c_attr_node)
            # 将原条件结点c_node替换为扩展后子树subtree
            parent_of_c = c_node.parent
            parent_of_c.children[0] = subtree
            # 记录已扩展条件
            self.traversed.append(c)
            # 尝试在初始状态运行行为树
            val, obj = bt.tick(start)
            canrun = False
            if val == 'success' or val == 'running':
                canrun = True
        return bt



    def run_algorithm(self, start, goal, actions):
        # goal_ls = goal.replace(" ", "")
        # goal_ls = goal_ls.split("|")
        self.bt = ControlBT(type='cond')
        subtree = ControlBT(type='?')
        if len(goal) > 1:
            for g in goal:
                print("goal",g)
                bt_sel_tree = self.run_algorithm_selTree(start, g, actions)
                print("bt_sel_tree.children",bt_sel_tree.children)
                # print(bt_sel_tree.children[0])
                subtree.add_child([copy.deepcopy(bt_sel_tree.children[0])])
            self.bt.add_child([subtree])
        else:
            self.bt = self.run_algorithm_selTree(start, goal[0], actions)
        return True

    def print_solution(self):
        print(len(self.nodes))
        # for i in self.nodes:
        #     if isinstance(i,Node):
        #         print (i.content)
        #     else:
        #         print (i)

    # 树的dfs
    def dfs_btml(self,parnode,is_root=False):
        for child in parnode.children:
            if isinstance(child, Leaf):
                if child.type == 'cond':

                    if is_root and len(child.content) > 1:
                        # 把多个 cond 串起来
                        self.btml_string += "sequence{\n"
                        self.btml_string += "cond "
                        c_set_str = '\n cond '.join(map(str, child.content)) + "\n"
                        self.btml_string += c_set_str
                        self.btml_string += '}\n'
                    else:
                        self.btml_string += "cond "
                        c_set_str = '\n cond '.join(map(str, child.content)) + "\n"
                        self.btml_string += c_set_str

                elif child.type == 'act':
                    if '(' not in child.content.name:
                        self.btml_string += 'act ' + child.content.name + "()\n"
                    else:
                        self.btml_string += 'act ' + child.content.name + "\n"
            elif isinstance(child, ControlBT):
                if child.type == '?':
                    self.btml_string += "selector{\n"
                    self.dfs_btml(parnode=child)
                elif child.type == '>':
                    self.btml_string += "sequence{\n"
                    self.dfs_btml( parnode=child)
                self.btml_string += '}\n'


    def dfs_btml_indent(self, parnode, level=0, is_root=False):
        indent = " " * (level * 4)  # 4 spaces per indent level
        for child in parnode.children:
            if isinstance(child, Leaf):

                if child.type == 'cond':
                    if not is_root and len(child.content) > 1:
                        # 把多个 cond 串起来
                        self.btml_string += " " * (level * 4) + "sequence\n"
                        for c in child.content:
                            self.btml_string += " " * ((level + 1) * 4) + "cond " + str(c) + "\n"
                    else:
                        # 直接添加cond及其内容，不需要特别处理根节点下多个cond的情况
                        # self.btml_string += indent + "cond " + ', '.join(map(str, child.content)) + "\n"
                        # 对每个条件独立添加，确保它们各占一行
                        for c in child.content:
                            self.btml_string += indent + "cond " + str(c) + "\n"
                elif child.type == 'act':
                    # 直接添加act及其内容
                    self.btml_string += indent + 'act ' + child.content.name + "\n"
            elif isinstance(child, ControlBT):
                if child.type == '?':
                    self.btml_string += indent + "selector\n"
                    self.dfs_btml_indent(child, level + 1)  # 增加缩进级别
                elif child.type == '>':
                    self.btml_string += indent + "sequence\n"
                    self.dfs_btml_indent(child, level + 1)  # 增加缩进级别

    def get_btml(self, use_braces=True):

        if use_braces:
            self.btml_string = "selector\n"
            self.dfs_btml_indent(self.bt.children[0], 1, is_root=True)
            return self.btml_string
        else:
            self.btml_string = "selector{\n"
            self.dfs_btml(self.bt.children[0], is_root=True)
            self.btml_string += '}\n'
        return self.btml_string


    def save_btml_file(self,file_name):
        self.btml_string = "selector{\n"
        self.dfs_btml(self.bt.children[0])
        self.btml_string += '}\n'
        with open(f'./{file_name}.btml', 'w') as file:
            file.write(self.btml_string)
        return self.btml_string


# 所对比的基准算法，具体扩展细节有差异



if __name__ == '__main__':
    random.seed(1)
    # 设置生成规划问题集的超参数：文字数、解深度、迭代次数
    literals_num = 10
    depth = 10
    iters = 10
    total_tree_size = []
    total_action_num = []
    total_state_num = []
    total_steps_num = []
    # fail_count=0
    # danger_count=0
    success_count = 0
    failure_count = 0
    planning_time_total = 0.0
    # 实验1000次
    for count in range(0, 1000):
        # 生成一个规划问题，包括随机的状态和行动，以及目标状态
        states = []
        actions = []
        start = generate_random_state(literals_num)
        state = start
        states.append(state)
        # print (state)
        for i in range(0, depth):
            a = Action()
            a.generate_from_state(state, literals_num)
            if not a in actions:
                actions.append(a)
            state = state_transition(state, a)
            if state in states:
                pass
            else:
                states.append(state)
                # print(state)

        goal = states[-1]
        state = start
        for i in range(0, iters):
            a = Action()
            a.generate_from_state(state, literals_num)
            if not a in actions:
                actions.append(a)
            state = state_transition(state, a)
            if state in states:
                pass
            else:
                states.append(state)
            state = random.sample(states, 1)[0]
        # 选择测试本文算法btalgorithm，或对比算法weakalgorithm
        algo = BTalgorithm()
        # algo = Weakalgorithm()
        start_time = time.time()
        if algo.run_algorithm(start, goal, list(actions)):  # 运行算法，规划后行为树为algo.bt
            total_tree_size.append(algo.bt.count_size() - 1)
        else:
            print("error")
        end_time = time.time()
        planning_time_total += (end_time - start_time)

        # 开始从初始状态运行行为树，测试
        state = start
        steps = 0
        val, obj = algo.bt.tick(state)  # tick行为树，obj为所运行的行动
        while val != 'success' and val != 'failure':  # 运行直到行为树成功或失败
            state = state_transition(state, obj)
            val, obj = algo.bt.tick(state)
            if (val == 'failure'):
                print("bt fails at step", steps)
            steps += 1
            if (steps >= 500):  # 至多运行500步
                break
        if not goal <= state:  # 错误解，目标条件不在执行后状态满足
            # print ("wrong solution",steps)
            failure_count += 1

        else:  # 正确解，满足目标条件
            # print ("right solution",steps)
            success_count += 1
            total_steps_num.append(steps)
        algo.clear()
        total_action_num.append(len(actions))
        total_state_num.append(len(states))
    print(success_count, failure_count)  # 算法成功和失败次数

    print(np.mean(total_tree_size), np.std(total_tree_size, ddof=1))  # 1000次测试树大小
    print(np.mean(total_steps_num), np.std(total_steps_num, ddof=1))
    print(np.mean(total_state_num))  # 1000次问题的平均状态数
    print(np.mean(total_action_num))  # 1000次问题的平均行动数
    print(planning_time_total, planning_time_total / 1000.0)

    # print(total_state_num)

    # casestudy begin 对应论文的case study，包含三个行动的移动机械臂场景

    actions = []
    a = Action(name='movebtob')
    a.pre = {1, 2}
    a.add = {3}
    a.del_set = {1, 4}
    actions.append(a)
    a = Action(name='moveatob')
    a.pre = {1}
    a.add = {5, 2}
    a.del_set = {1, 6}
    actions.append(a)
    a = Action(name='moveatoa')
    a.pre = {7}
    a.add = {8, 2}
    a.del_set = {7, 6}
    actions.append(a)

    start = {1, 7, 4, 6}
    goal = {3}
    algo = BTalgorithm()
    algo.clear()
    algo.run_algorithm(start, goal, list(actions))
    state = start
    steps = 0
    val, obj = algo.bt.tick(state)
    while val != 'success' and val != 'failure':
        state = state_transition(state, obj)
        print(obj.name)
        val, obj = algo.bt.tick(state)
        if (val == 'failure'):
            print("bt fails at step", steps)
        steps += 1
    if not goal <= state:
        print("wrong solution", steps)
    else:
        print("right solution", steps)
    # algo.bt.print_nodes()
    print(algo.bt.count_size() - 1)
    algo.clear()

# case study end

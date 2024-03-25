import copy
import random
import heapq
import re
from btgym.algos.bt_autogen.behaviour_tree import Leaf,ControlBT
from btgym.algos.bt_autogen.Action import Action

class CondActPair:
    def __init__(self, cond_leaf,act_leaf):
        self.cond_leaf = cond_leaf
        self.act_leaf = act_leaf
    def __lt__(self, other):
        # 定义优先级比较：按照 cost 的值来比较
        return self.act_leaf.mincost < other.act_leaf.mincost


# def conflict(c):
#     have_at = False
#     for str in c:
#         if 'Not' not in str and 'RobotNear' in str:
#             if have_at:
#                 return True
#             have_at = True
#
#     Holding = False
#     HoldingNothing = False
#     for str in c:
#         if 'Not ' not in str and 'Holding(Nothing)' in str: # 注意 'Not ' in 'Nothing'
#             HoldingNothing = True
#         if 'Not' not in str and 'Holding(Nothing)' not in str and 'Holding' in str:
#             if Holding:
#                 return True
#             Holding = True
#         if HoldingNothing and Holding:
#             return True
#     return False



class OptBTExpAlgorithm:
    def __init__(self,verbose=False,lit_act_dic=None):
        self.bt = None
        self.nodes=[]
        self.traversed=[]
        self.mounted=[]
        self.conditions=[]
        self.conditions_index=[]
        self.verbose=verbose
        self.goal=None
        self.bt_merge = True
        self.lit_act_dic = lit_act_dic

    def clear(self):
        self.bt = None
        self.nodes = []
        self.traversed = [] #存cond
        self.expanded = [] #存整个
        self.conditions = []
        self.conditions_index = []
        self.traversed_state_num=0
        self.fot_times = 0
        self.expand_conds=0
        self.tree_size=0
        self.bt_without_merge = None
        self.subtree_count=1

    #运行规划算法，从初始状态、目标状态和可用行动，计算行为树self.bt
    # def run_algorithm(self,goal,actions,scene):
    def run_algorithm_selTree(self, start, goal, actions,merge_time=99999999):


        self.traversed_state_num=0

        self.goal = goal
        if self.verbose:
            print("\n算法开始！")
        bt = ControlBT(type='cond')
        # 初始行为树只包含目标条件
        gc_node = Leaf(type='cond', content=goal, mincost=0) # 为了统一，都成对出现
        ga_node = Leaf(type='act', content=None, mincost=0)
        subtree = ControlBT(type='?')
        subtree.add_child([gc_node])  # 子树首先保留所扩展结
        self.expand_conds+=1
        bt.add_child([subtree])
        cond_anc_pair = CondActPair(cond_leaf=gc_node,act_leaf=ga_node)

        heapq.heappush(self.nodes, cond_anc_pair)
        self.expanded.append(goal)
        self.traversed_state_num += 1

        self.traversed = [goal] # the set of expanded conditions
        min_cost = float('inf')

        if goal <= start:
            self.bt_without_merge = bt
            return bt, 0

        while len(self.nodes)!=0:

            self.fot_times+=1

            #  Find the condition for the shortest cost path
            # ======================== Next Goal ============================ #
            min_cost = float ('inf')
            pair_node = heapq.heappop(self.nodes)

            if self.verbose:
                print("选择扩展条件结点：",pair_node.cond_leaf.content)
            # Update self.nodes and self.traversed
            c = pair_node.cond_leaf.content  # 子树所扩展结点对应的条件（一个文字的set）
            # Mount the action node and extend BT. T = Eapand(T,c,A(c))



            if c!=goal:
                if c!=set():
                    sequence_structure = ControlBT(type='>')
                    sequence_structure.add_child(
                        [pair_node.cond_leaf, pair_node.act_leaf])
                    subtree.add_child([copy.deepcopy(sequence_structure)])  # subtree 是回不断变化的，它的父亲是self.bt
                    # self.expanded.append(copy.deepcopy(pair_node))
                    # self.expanded.append(pair_node.cond_leaf.content)
                    #
                    if c <= start:
                        if self.bt_merge:
                            # bt = self.merge_adjacent_conditions_stack(bt)
                            bt = self.merge_adjacent_conditions_stack_time(bt,merge_time=merge_time)
                        return bt, min_cost
                else:
                    subtree.add_child([copy.deepcopy(pair_node.act_leaf)])
                self.expand_conds += 1
                self.expanded.append(c)
                if c <= start:
                    self.tree_size = self.bfs_cal_tree_size_subtree(bt)
                    self.bt_without_merge = bt
                    if self.bt_merge:
                        # bt = self.merge_adjacent_conditions_stack(bt)
                        bt = self.merge_adjacent_conditions_stack_time(bt,merge_time=merge_time)
                    return bt, min_cost



                if self.verbose:
                    print("完成扩展 a_node= %s,对应的新条件 c_attr= %s,mincost=%d" \
                          % (cond_anc_pair.act_leaf.content.name, cond_anc_pair.cond_leaf.content,
                             cond_anc_pair.cond_leaf.mincost))

            if self.verbose:
                print("遍历所有动作, 寻找符合条件的动作")
            # 遍历所有动作, 寻找符合条件的动作
            current_mincost = pair_node.cond_leaf.mincost # 当前的最短路径是多少
            # ======================== End Next Goal ============================ #

            # ====================== Action Trasvers ============================ #
            traversed_current = []

            # act_tmp_set = set()
            # for lit in c:
            #     act_tmp_set |= self.lit_act_dic[lit]

            # for i in act_tmp_set:
            for i in range(0, len(actions)):


                if not c & ((actions[i].pre | actions[i].add) - actions[i].del_set) <= set()  :
                    if (c - actions[i].del_set) == c:
                        if self.verbose:
                            print("———— 满足条件可以扩展")
                        c_attr = (actions[i].pre | c) - actions[i].add

                        # if conflict(c_attr):
                        #     if self.verbose:
                        #         print("———— 冲突：动作 %s， 条件 %s"% (actions[i].name,c_attr))
                        #     continue

                        # 剪枝操作,现在的条件是以前扩展过的条件的超集
                        valid = True

                        for j in self.expanded:
                            if j <= c_attr:
                                valid = False
                                break

                        # tmp_heap = list(self.nodes)
                        # # tmp_heap = copy.deepcopy(self.nodes)
                        # while tmp_heap:  # 剪枝操作 self.expanded?
                        #     cond_anc_pair = heapq.heappop(tmp_heap)
                        #     j = cond_anc_pair.cond_leaf.content
                        #     if j <= c_attr:
                        #         if cond_anc_pair.cond_leaf.mincost < current_mincost + actions[i].cost:
                        #             valid = False
                        #             break


                        if valid:
                            # c_attr_string = "".join(sorted(list(c_attr)))
                            c_attr_node = Leaf(type='cond', content=c_attr, mincost=current_mincost + actions[i].cost)
                            a_attr_node = Leaf(type='act', content=actions[i], mincost=current_mincost + actions[i].cost)
                            cond_anc_pair = CondActPair(cond_leaf=c_attr_node, act_leaf=a_attr_node)
                            # heapq.heappush(self.nodes, copy.deepcopy(cond_anc_pair))
                            heapq.heappush(self.nodes, cond_anc_pair)


                            self.traversed_state_num+=1
                            traversed_current.append(c_attr)
                            # 把符合条件的动作节点都放到列表里
                            if self.verbose:
                                print("———— -- %s 符合条件放入列表,对应的c为 %s" % (actions[i].name,c_attr))

            # print(len(traversed_current))
            self.traversed.extend(traversed_current)
            # ====================== End Action Trasvers ============================ #
        self.tree_size = self.bfs_cal_tree_size_subtree(bt)
        self.bt_without_merge = bt
        if self.bt_merge:
            # bt = self.merge_adjacent_conditions_stack(bt)
            bt = self.merge_adjacent_conditions_stack_time(bt,merge_time=merge_time)
        if self.verbose:
            print("算法结束！\n")
        return bt,min_cost


    def run_algorithm(self, start, goal, actions,merge_time=3):
        self.bt = ControlBT(type='cond')
        subtree = ControlBT(type='?')

        subtree_with_costs_ls=[]

        self.subtree_count = len(goal)

        if len(goal) > 1:
            for g in goal:
                bt_sel_tree,mincost = self.run_algorithm_selTree(start, g, actions)
                subtree_with_costs_ls.append((bt_sel_tree,mincost))
            # 要排个序再一次add
            sorted_trees = sorted(subtree_with_costs_ls, key=lambda x: x[1])
            for tree,cost in sorted_trees:
                subtree.add_child([tree.children[0]])
            self.bt.add_child([subtree])
        else:
            self.bt,mincost = self.run_algorithm_selTree(start, goal[0], actions,merge_time=merge_time)
        return True


    def merge_subtree(self,merge_time):

        self.bt_aftermerge = ControlBT(type='cond')
        subtree = ControlBT(type='?')

        if self.subtree_count > 1:
            for i in range(self.subtree_count):
                bt_sel_tree = self.bt.children[0].children[i]

                bt_sel_tree_m = ControlBT(type='cond')
                bt_sel_tree_m.add_child([bt_sel_tree])

                bt_sel_tree_m = self.merge_adjacent_conditions_stack_time(copy.deepcopy(bt_sel_tree_m),merge_time=merge_time)
                subtree.add_child([bt_sel_tree_m.children[0]])
            self.bt_aftermerge.add_child([subtree])
        else:
            self.bt_aftermerge = self.merge_adjacent_conditions_stack_time(copy.deepcopy(self.bt),merge_time=merge_time)
        return self.bt_aftermerge

    def run_algorithm_test(self, start, goal, actions):
        self.bt,mincost = self.run_algorithm_selTree(start, goal, actions)
        return True


    def merge_adjacent_conditions_stack_time(self,bt_sel,merge_time=9999999):

        merge_time = min(merge_time,500)

        # 只针对第一层合并，之后要考虑层层递归合并
        bt = ControlBT(type='cond')
        sbtree = ControlBT(type='?')
        # gc_node = Leaf(type='cond', content=self.goal, mincost=0)  # 为了统一，都成对出现
        # sbtree.add_child([copy.deepcopy(gc_node)])  # 子树首先保留所扩展结
        bt.add_child([sbtree])

        parnode = bt_sel.children[0]
        stack=[]
        time_stack=[]
        for child in parnode.children:
            if isinstance(child, ControlBT) and child.type == '>':
                if stack==[]:
                    stack.append(child)
                    time_stack.append(0)
                    continue
                # 检查合并的条件，前面一个的条件包含了后面的条件，把包含部分提取出来
                last_child = stack[-1]
                last_time = time_stack[-1]



                if last_time<merge_time and isinstance(last_child, ControlBT) and last_child.type == '>':
                    set1 = last_child.children[0].content
                    set2 = child.children[0].content
                    inter = set1 & set2

                    # print("merge time:", last_time,set1,set2)

                    if inter!=set():
                        c1 = set1-set2
                        c2 = set2-set1
                        inter_node = Leaf(type='cond', content=inter)
                        c1_node = Leaf(type='cond', content=c1)
                        c2_node = Leaf(type='cond', content=c2)
                        a1_node = last_child.children[1]
                        a2_node = child.children[1]


                        # set1<=set2,此时set2对应的动作永远不会执行
                        if (c1==set() and isinstance(last_child.children[1], Leaf) and isinstance(child.children[1], Leaf) \
                               and isinstance(last_child.children[1].content, Action) and isinstance(child.children[1].content, Action)):
                            continue

                        # 再写一个特殊情况处理，三个结点动作last 遇到 两个结点 且动作相同
                        if len(last_child.children)==3 and \
                            isinstance(last_child.children[2], Leaf) and isinstance(child.children[1], Leaf) \
                                and isinstance(last_child.children[2].content, Action) and isinstance( child.children[1].content, Action) \
                                and last_child.children[2].content.name == child.children[1].content.name \
                                and c1==set() and c2!=set():
                                    last_child.children[1].add_child([c2_node])
                                    continue
                        elif len(last_child.children)==3:
                            stack.append(child)
                            time_stack.append(0)
                            continue

                        # 判断动作相不相同
                        if isinstance(last_child.children[1], Leaf) and isinstance(child.children[1], Leaf) \
                            and isinstance(last_child.children[1].content, Action) and isinstance(child.children[1].content, Action) \
                                and last_child.children[1].content.name == child.children[1].content.name:

                            if c2==set():
                                tmp_tree = ControlBT(type='>')
                                tmp_tree.add_child(
                                    [inter_node, a1_node])
                            else:
                                _sel = ControlBT(type='?')
                                _sel.add_child([c1_node, c2_node])
                                tmp_tree = ControlBT(type='>')
                                tmp_tree.add_child(
                                    [inter_node, _sel,a1_node])
                        else:
                            if c1 == set():
                                seq1 = last_child.children[1]
                            else:
                                seq1 = ControlBT(type='>')
                                seq1.add_child([c1_node, a1_node])

                            if c2 == set():
                                seq2 = child.children[1]
                            else:
                                seq2 = ControlBT(type='>')
                                seq2.add_child([c2_node, a2_node])
                            sel = ControlBT(type='?')
                            sel.add_child([seq1, seq2])
                            tmp_tree = ControlBT(type='>')
                            tmp_tree.add_child(
                                [inter_node,sel])

                        stack.pop()
                        time_stack.pop()
                        stack.append(tmp_tree)
                        time_stack.append(last_time+1)

                    else:
                        stack.append(child)
                        time_stack.append(0)
                else:
                    stack.append(child)
                    time_stack.append(0)
            else:
                stack.append(child)
                time_stack.append(0)

        for tree in stack:
            sbtree.add_child([tree])
        bt_sel = bt
        return bt_sel





    def print_solution(self,without_merge=False):
        print("========= BT ==========")  # 树的bfs遍历
        nodes_ls = []
        if without_merge==True:
            nodes_ls.append(self.bt_without_merge)
        else:
            nodes_ls.append(self.bt)
        while len(nodes_ls) != 0:
            parnode = nodes_ls[0]
            print("Parrent:", parnode.type)
            for child in parnode.children:
                if isinstance(child, Leaf):
                    print("---- Leaf:", child.content)
                elif isinstance(child, ControlBT):
                    print("---- ControlBT:", child.type)
                    nodes_ls.append(child)
            print()
            nodes_ls.pop(0)
        print("========= BT ==========\n")

    # 返回所有能到达目标状态的初始状态
    def get_all_state_leafs(self):
        state_leafs=[]

        nodes_ls = []
        nodes_ls.append(self.bt)
        while len(nodes_ls) != 0:
            parnode = nodes_ls[0]
            for child in parnode.children:
                if isinstance(child, Leaf):
                    if child.type == "cond":
                        state_leafs.append(child.content)
                elif isinstance(child, ControlBT):
                    nodes_ls.append(child)
            nodes_ls.pop(0)

        return state_leafs


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

    def get_btml(self,use_braces=True):

        if use_braces:
            self.btml_string = "selector\n"
            self.dfs_btml_indent(self.bt.children[0], 1, is_root=True)
            return self.btml_string
        else:
            self.btml_string = "selector{\n"
            self.dfs_btml(self.bt.children[0],is_root=True)
            self.btml_string += '}\n'
        return self.btml_string






    # def dfs_btml_many_act(self, parnode, is_root=False):
    #     for child in parnode.children:
    #         if isinstance(child, Leaf):
    #             if child.type == 'cond':
    #
    #                 # if is_root and len(child.content) > 1:
    #                 if is_root and len(child.content) > 1:
    #                     # 把多个 cond 串起来
    #                     self.btml_string += "sequence{\n"
    #                     self.btml_string += "cond "
    #                     c_set_str = '\n cond '.join(map(str, child.content)) + "\n"
    #                     self.btml_string += c_set_str
    #                     self.btml_string += '}\n'
    #                 # elif is_root:
    #                 else:
    #                     self.btml_string += "cond "
    #                     c_set_str = '\n cond '.join(map(str, child.content)) + "\n"
    #                     self.btml_string += c_set_str
    #                 # else:
    #                 #     c_set_positive = {item for item in child.content if 'Not' not in item}
    #                 #     self.btml_string += "cond "
    #                 #     c_set_str = '\n cond '.join(map(str, c_set_positive)) + "\n"
    #                 #     self.btml_string += c_set_str
    #
    #
    #             elif child.type == 'act':
    #
    #                 # child.content.name = re.sub(r'\d+', '', child.content.name)
    #                 child.content.name = re.sub(r'\)\d+', ')', child.content.name)
    #                 if '(' not in child.content.name:
    #                     self.btml_string += 'act ' + child.content.name + "()\n"
    #                 else:
    #                     self.btml_string += 'act ' + child.content.name + "\n"
    #         elif isinstance(child, ControlBT):
    #             if child.type == '?':
    #                 self.btml_string += "selector{\n"
    #                 if len(child.children)>2:
    #                     self.dfs_btml_many_act(parnode=child, is_root=True)
    #                 else:
    #                     self.dfs_btml_many_act(parnode=child)
    #             elif child.type == '>':
    #                 self.btml_string += "sequence{\n"
    #                 self.dfs_btml_many_act(parnode=child)
    #             self.btml_string += '}\n'
    #
    # def get_btml_many_act(self):
    #     self.btml_string = "selector{\n"
    #     self.dfs_btml_many_act(self.bt.children[0],is_root=True)
    #     self.btml_string += '}\n'
    #     return self.btml_string

    def save_btml_file(self,file_name):
        self.btml_string = "selector{\n"
        self.dfs_btml(self.bt.children[0])
        self.btml_string += '}\n'
        with open(f'./{file_name}.btml', 'w') as file:
            file.write(self.btml_string)
        return self.btml_string


    def bfs_cal_tree_size(self):
        from collections import deque
        queue = deque([self.bt.children[0]])

        count = 0
        while queue:
            current_node = queue.popleft()
            count += 1
            for child in current_node.children:
                if isinstance(child, Leaf):
                    count += 1
                else:
                    queue.append(child)
        return count

    def bfs_cal_tree_size_subtree(self,bt):
        from collections import deque
        queue = deque([bt.children[0]])

        count = 0
        while queue:
            current_node = queue.popleft()
            count += 1
            for child in current_node.children:
                if isinstance(child, Leaf):
                    count += 1
                else:
                    queue.append(child)
        return count
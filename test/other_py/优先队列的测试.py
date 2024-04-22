import heapq

# 优先队列
nodes = []

# 辅助类
class CondActPair:
    def __init__(self, cond_leaf, act_leaf):
        self.cond_leaf = cond_leaf
        self.act_leaf = act_leaf


    def __lt__(self, other):
        # 定义优先级比较：按照 cost 的值来比较
        return self.act_leaf < other.act_leaf

# 添加一些示例节点到优先队列中

heapq.heappush(nodes, CondActPair(cond_leaf="a", act_leaf=10))
heapq.heappush(nodes, CondActPair(cond_leaf="b", act_leaf=20))
heapq.heappush(nodes, CondActPair(cond_leaf="c", act_leaf=30))

# for node in nodes:
#     if node.cond_leaf=="a":
#         node.act_leaf =88
#     elif node.cond_leaf=="b":
#         node.act_leaf = 40
temp_nodes=[]
while nodes:
    node = heapq.heappop(nodes)
    if node.cond_leaf == "a":
        node.act_leaf = 88
    elif node.cond_leaf == "b":
        node.act_leaf = 40
    elif node.cond_leaf == "c":
        node.act_leaf = 1
    heapq.heappush(temp_nodes, node)
while nodes!=[]:
    node = heapq.heappop(nodes)
    print("node:",node.cond_leaf,"  ",node.act_leaf)
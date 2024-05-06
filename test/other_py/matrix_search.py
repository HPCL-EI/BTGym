import numpy as np

# 定义初始状态向量，True表示状态为True，False表示状态为False
initial_state = np.array([True, False, True])

# 定义3个动作，每个动作由pre, add, del三个3维向量组成，构成一个3x3的矩阵
# 这里使用True/False来表示向量中各个元素的值
actions = np.array([
    # 动作1
    [[True, False, True],  # pre 条件
     [False, True, False],  # add 效果
     [False, False, True]], # del 效果
    # 动作2
    [[False, True, False],  # pre
     [True, False, True],   # add
     [True, False, False]], # del
    # 动作3
    [[False, False, True],  # pre
     [True, True, False],   # add
     [False, True, False]]  # del
])

# 计算执行每个动作后的状态
final_states = []

for action in actions:
    pre, add, del_effect = action
    # 检查pre条件是否被满足（即初始状态与pre向量相匹配）
    if np.all(initial_state == pre) or np.all(~pre):
        # 应用add效果（逻辑或），并且移除del效果（逻辑非）
        new_state = (initial_state | add) & ~del_effect
    else:
        # 如果pre条件不满足，则状态不变
        new_state = initial_state
    final_states.append(new_state)

# 将最终状态组装成一个3维张量
final_states_tensor = np.array(final_states)

print(final_states_tensor)

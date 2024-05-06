# 假设的观察数据，每个观察包括初始状态、执行的动作和最终状态
observations = [
    ({'Box': 'Closed'}, 'OpenBox', {'Box': 'Opened'}),
    ({'Box': 'Opened'}, 'CloseBox', {'Box': 'Closed'})
]


# 学习动作模型
def learn_action_models(observations):
    action_models = {}

    for initial_state, action, final_state in observations:
        # 如果动作模型中还没有这个动作，则初始化它
        if action not in action_models:
            action_models[action] = {'preconditions': set(), 'effects': set()}

        # 根据观察更新前提条件和效果
        preconditions = action_models[action]['preconditions']
        effects = action_models[action]['effects']

        # 对于每个观察，我们假设初始状态的所有条件都是前提条件
        # 并且最终状态中有变化的条件是效果
        for state, value in initial_state.items():
            preconditions.add(f"{state} == '{value}'")

            if state in final_state and final_state[state] != value:
                effects.add(f"{state} = '{final_state[state]}'")
            elif state not in final_state:
                effects.add(f"del {state}")

    return action_models


# 学习动作模型
action_models = learn_action_models(observations)

# 打印学习到的动作模型
for action, model in action_models.items():
    print(f"Action: {action}")
    print(f"  Preconditions: {model['preconditions']}")
    print(f"  Effects: {model['effects']}\n")

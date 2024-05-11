import pandas as pd
import btgym
import time
from btgym import BehaviorTree
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction
from btgym.utils import ROOT_PATH
from btgym.utils.read_dataset import read_dataset
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process, act_format_records


def load_result_csv(file_name):
    # Load the results from the previously saved CSV file
    # file_name = "llm_40.csv"
    results_df = pd.read_csv(file_name)
    results = results_df.to_dict(orient='records')  # Convert DataFrame back to list of dictionaries
    # Check that the data is successfully loaded
    print(f"Loaded {len(results)} results from '{file_name}'")


def load_dataset(data_path):
    data1 = read_dataset(data_path)
    len_data = len(data1)
    print(f"导入 {len_data} 条数据")
    return data1


# 导入大模型的结果
# dataset = load_result_csv("llm_40_just_goal_gpt4.csv")
# data = dataset

# 导入数据集 真实值
dataset1 = load_dataset(f"{ROOT_PATH}/../test/dataset/data1_env1_40_big_test.txt")

results = []
start_time = time.time()
for id, d in enumerate(dataset1[1:6]): # 5可以
    print("\n== ID:", id, "  ", d['Instruction'])

    goal_str = ' & '.join(d["Goals"])
    # goal_str="IsOn_pear_kitchentable & IsOn_bananas_kitchentable"
    goal_set = goal_transfer_str(goal_str)
    print("goal_set:", goal_set)

    priority_act_ls = act_str_process(d["Optimal Actions"])
    print("priority_act_ls:", priority_act_ls)

    key_pred = d['Vital Action Predicates']
    key_obj = d['Vital Objects']

    result_entry = {
        'id': id,
        'Instruction': d['Instruction'],
        'Goals': d['Goals'],
        'Optimal Actions': d['Optimal Actions'],
        'Vital Action Predicates': d['Vital Action Predicates'],
        'Vital Objects': d['Vital Objects']
    }

    env = btgym.make("VHT-PutMilkInFridge")
    cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
    cur_cond_set |= {f'IsClose({arg})' for arg in VHTAction.CAN_OPEN}
    cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHTAction.HAS_SWITCH}
    cur_cond_set |= {f'IsUnplugged({arg})' for arg in VHTAction.HAS_PLUG}

    for use_priority_act in [False, True]:
        # mode = 'small-predicate-objs'
        mode = 'big'
        # use_priority_act=True
        print('-----------------------------------------------------')
        print(f"mode = {mode}, use_priority_act = {use_priority_act}")

        algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set, \
                              priority_act_ls=priority_act_ls, key_predicates=key_pred, key_objects=key_obj, \
                              selected_algorithm="opt", mode=mode, \
                              llm_reflect=False, use_priority_act=use_priority_act,time_limit=None)

        start_time = time.time()
        algo.process(goal_set)
        end_time = time.time()

        ptml_string, cost, expanded_num = algo.post_process()
        print("Expanded Conditions: ", expanded_num)
        planning_time_total = (end_time - start_time)
        print("planning_time_total:", planning_time_total)
        print("cost_total:", cost)

        time_limit_exceeded = algo.algo.time_limit_exceeded

        # Simulation and test
        print("\n================ ")
        goal = goal_set[0]
        state = cur_cond_set
        error, state, act_num, current_cost, record_act_ls = algo.execute_bt(goal, state, verbose=False)
        print(f"Executed {act_num - 1} action steps")
        print("current_cost:", current_cost)
        print("================ ")

        # Add the results to the entry with appropriate prefixes
        prefix = "P_" if use_priority_act else "NP_"
        result_entry[f'{prefix}Timeout'] = 'Timeout' if time_limit_exceeded == True else ''
        result_entry[f'{prefix}err'] = 'error' if error==True else ''
        result_entry[f'{prefix}exp'] = expanded_num
        result_entry[f'{prefix}time'] = planning_time_total
        result_entry[f'{prefix}cost'] = cost
        result_entry[f'{prefix}act'] = record_act_ls

    results.append(result_entry)

    # if id>=0 and id % 5 ==0:

    df = pd.DataFrame(results)
    # Specify the exact column order
    columns = [
        'id', 'Instruction', 'Goals', 'Optimal Actions',
        'Vital Action Predicates', 'Vital Objects',
        'NP_err', 'P_err',
        'NP_exp', 'P_exp',
        'NP_time', 'P_time',
        'NP_cost', 'P_cost',
        'NP_act', 'P_act'
    ]
    df = df[columns]  # Reorder according to the specified columns
    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()).replace("-", "").replace(":", "")
    df.to_csv(f"BT_40_time={time_str}.csv", index=False)
    print(f"Results have been saved to BT_40_time={time_str}.csv")


end_time = time.time()
time_total = (end_time - start_time)
print("Total time:", time_total)

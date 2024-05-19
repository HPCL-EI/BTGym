from btgym.algos.llm_client.llm_ask_tools import extract_llm_from_instr_goal, extract_llm_from_reflect, \
    convert_conditions, format_example
from tools import execute_algorithm, load_dataset, setup_default_env
from tools import find_from_small_act,load_dataset_and_cost
import time
import random
import numpy as np
import pandas as pd
import btgym
from btgym.utils import ROOT_PATH
from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process, act_format_records
from btgym.utils.tools import collect_action_nodes,extract_objects

from btgym.utils.read_dataset import read_dataset
# Set random seed
random_seed = 0
np.random.seed(random_seed)
random.seed(random_seed)

# Initialize the LLM
llm = LLMGPT3()


def convert_set_to_str(string_set):
    return ", ".join(f'\"{s}\"' for s in string_set)

def get_or_default(value, default):
    return value if value is not None else default


def perform_test(env, chosen_goal, database_index_path, reflect_time=0, train=False, choose_database=True):
    cur_cond_set = setup_default_env()[1]
    priority_act_ls, llm_key_pred, llm_key_obj, messages, distances = \
        extract_llm_from_instr_goal(llm, default_prompt_file, 1, chosen_goal, verbose=False,
                                    choose_database=choose_database, database_index_path=database_index_path)
    if priority_act_ls != None:
        _priority_act_ls, pred, obj = act_format_records(priority_act_ls)
        key_predicates = list(set(llm_key_pred + pred))
        key_objects = list(set(llm_key_obj + obj))
        # goal里的目标也要加进去
        algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                              priority_act_ls=priority_act_ls, key_predicates=key_predicates,
                              key_objects=key_objects,
                              selected_algorithm="opt", mode="small-predicate-objs",
                              llm_reflect=False, time_limit=10,
                              heuristic_choice=0)
        goal_set = goal_transfer_str(' & '.join(chosen_goal))
        expanded_num, planning_time_total, cost, error, act_num, current_cost, record_act_ls = \
            execute_algorithm(algo, goal_set, cur_cond_set)
        time_limit_exceeded = algo.algo.time_limit_exceeded
        success = not error and not time_limit_exceeded
    else:
        success = False
        goal_set, key_predicates, key_objects = None, None, None

    if not success and train:
        # 搜索小动作空间得到一个解
        success, _priority_act_ls, key_predicates, key_objects, cost, priority_act_ls, key_predicates, key_objects, \
            act_num, error, time_limit_exceeded, current_cost, expanded_num, planning_time_total = find_from_small_act(
            chosen_goal)

    if choose_database:
        if priority_act_ls is None or llm_key_pred is None or llm_key_obj is None:
            return False, np.mean(
                distances), None, None, None, None, priority_act_ls, None, None, \
                None, None, None, None, None, None

        return success, np.mean(
            distances), _priority_act_ls, key_predicates, key_objects, cost, priority_act_ls, key_predicates, key_objects, \
            act_num, error, time_limit_exceeded, current_cost, expanded_num, planning_time_total
    else:
        if priority_act_ls is None or llm_key_pred is None or llm_key_obj is None:
            return False, 0, None, None, None, None, priority_act_ls, None, None, \
                None, None, None, None, None, None

        return success, 0, _priority_act_ls, key_predicates, key_objects, cost, priority_act_ls, key_predicates, key_objects, \
            act_num, error, time_limit_exceeded, current_cost, expanded_num, planning_time_total


# Function to validate a single goal
def validate_goal(env, chosen_goal, n, database_index_path=None, round_num=None,database_num=None, reflect_time=0,
                  choose_database=True):
    print(f"test:{n}", chosen_goal)

    test_result = perform_test(env, chosen_goal, database_index_path, reflect_time=reflect_time,
                               choose_database=choose_database)
    success, avg_similarity, _, key_predicates, key_objects, _, priority_act_ls, key_predicates, key_objects, \
        act_num, error, time_limit_exceeded, current_cost, expanded_num, planning_time_total = test_result

    fail=0
    while not success:
        fail+=1
        print("fail:",fail)

        # 在这里添加例子，加入到
        # 如果这里面把例子中的pred和obj也加进去
        # if nearest_examples != None:
        #     ex_preds = set()
        #     ex_objs = set()
        #     for ex in nearest_examples:
        #         ex_preds |= set(ex['value']['Vital Action Predicates'].replace(" ", "").split(","))
        #         ex_objs |= set(ex['value']['Vital Objects'].replace(" ", "").split(","))
        #     key_predicates = list(set(key_predicates) | ex_preds)
        #     key_objects = list(set(key_objects) | ex_objs)
        #
        #     pass


        # 疑问：动作空间如何每次计算？没加一个要遍历所有动作？加了物体又怎么判别呢
        # 感觉可以加5个example、10个example、20个example....

        # VD 实验， 简单、依赖、长序列、混合？ 成功率和expanded？ 要扩张吗
        # 感觉体现向量数据库学习过程，可以不扩张，单纯看 四种数据集上的学习能力


        # return {
        #     'round': round_num, 'id': n, 'goals': ' & '.join(chosen_goal), 'priority_act_ls': None,
        #     'key_predicates': None, 'key_objects': None, 'act_num': None, 'error': 'None',
        #     'time_limit_exceeded': 'None', 'current_cost': None, 'expanded_num': None, 'planning_time_total': None,
        #     'average_distance': None, 'database_size': database_num
        # }, False, avg_similarity

    print(f"\033[92mtest:{n} {chosen_goal} {act_num}\033[0m")
    return {
        'round': round_num, 'id': n, 'goals': ' & '.join(chosen_goal), 'priority_act_ls': priority_act_ls,
        'key_predicates': key_predicates, 'key_objects': key_objects, 'act_num': act_num, 'error': error,
        'time_limit_exceeded': time_limit_exceeded, 'current_cost': current_cost, 'expanded_num': expanded_num,
        'planning_time_total': planning_time_total, 'average_distance': avg_similarity, 'database_size': database_num
    }, success, avg_similarity



# ============= VH ================
# name = "vh"
# default_prompt_file = f"prompt_VH.txt"
# dataset = load_dataset_and_cost(f"vh_processed_data.txt")
# # env, _ = setup_default_env()
# from btgym.envs.virtualhome.exec_lib._base.VHAction import VHAction
# env = btgym.make("VH-PutMilkInFridge")
# cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
# cur_cond_set |= {f'IsClose({arg})' for arg in VHAction.CanOpenPlaces}
# cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHAction.HasSwitchObjects}
# big_actions = collect_action_nodes(env.behavior_lib)



# name = "RHS"
# default_prompt_file = f"prompt_VH.txt"
# dataset = read_dataset(f"RHS.txt")
# # dataset = load_dataset_and_cost(f"RHS.txt")
# # env, _ = setup_default_env()
# from btgym.envs.virtualhometextsmall.exec_lib._base.VHTAction import VHTAction as RHS
# env = btgym.make("VHT-Small")
# cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
# cur_cond_set |= {f'IsClose({arg})' for arg in RHS.CAN_OPEN}
# cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in RHS.HAS_SWITCH}
# cur_cond_set |= {f'IsUnplugged({arg})' for arg in RHS.HAS_PLUG}
# print(f"共收集到 {len(RHS.AllObject)} 个物体")



name = "RHS"
default_prompt_file = f"prompt_RHS_no_example.txt"
dataset = read_dataset(f"RHS_test_50.txt")
from btgym.envs.virtualhometextsmall.exec_lib._base.VHTAction import VHTAction
env = btgym.make("VHT-Small")
cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
cur_cond_set |= {f'IsClose({arg})' for arg in VHTAction.CAN_OPEN}
cur_cond_set |= {f'IsUnplugged({arg})' for arg in VHTAction.HAS_PLUG}
cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHTAction.HAS_SWITCH}
big_actions = collect_action_nodes(env.behavior_lib)



vaild_num = 11

# Initialize accumulators and counters
test_results = []
test_success_count = 0
total_similarity = 0
total_expanded_num = 0
total_planning_time_total = 0
total_current_cost = 0
total_cost_ratio = 0

# Dataframe to store metrics for each round
metrics_df = pd.DataFrame(columns=[
    "Test Success Rate",  "Average Expanded Num", "Average Planning Time Total", "Average Current Cost"
])

# ========================= 并行 ========================
# with concurrent.futures.ThreadPoolExecutor() as executor:
#     futures = [executor.submit(validate_goal, env, d['Goals'], database_index_path, round_num, n, database_num,
#                                reflect_time=reflect_time,choose_database=True) \
#                for n, d in enumerate(vaild_dataset_test)]
#     for future in concurrent.futures.as_completed(futures):
#         result, success, _ = future.result()
# ========================= 并行 ========================

# ========================= 串行========================
vaild_dataset = dataset[:vaild_num]
for n, d in enumerate(vaild_dataset):
    result, success, _ = validate_goal(env, d['Goals'], n, choose_database=False)
    test_results.append(result)
    if success:
        test_success_count += 1
    total_expanded_num += get_or_default(result.get('expanded_num'), 300)
    total_planning_time_total += get_or_default(result.get('planning_time_total'), 3)
    # total_cost_ratio += result.get('current_cost') / d['cost']
    # current_cost = get_or_default(result.get('current_cost'), 2000)
    # total_current_cost += current_cost
    total_cost_ratio += 0
    current_cost = 0
    total_current_cost += 0

# Calculate metrics
num_entries = len(vaild_dataset)
success_rate = test_success_count / num_entries
average_similarity = total_similarity / num_entries if num_entries else 0
average_expanded_num = total_expanded_num / num_entries if num_entries else 0
average_planning_time_total = total_planning_time_total / num_entries if num_entries else 0
average_cost_ratio = total_cost_ratio / num_entries if num_entries else 0
average_current_cost = total_current_cost / num_entries if num_entries else 0

# Append metrics to dataframe
round_metrics = pd.DataFrame([{
    "Test Success Rate": success_rate,
    "Average Expanded Num": average_expanded_num,
    "Average Planning Time Total": average_planning_time_total,
    "Average Cost Ratio": average_cost_ratio,
    "Average Current Cost": average_current_cost
}])

# Assuming metrics_df is initialized elsewhere in the complete code
metrics_df = pd.DataFrame()
metrics_df = pd.concat([metrics_df, round_metrics], ignore_index=True)

# Output metrics
print(f"Test Success Rate: {success_rate}")
print(f"Average Distance: {average_similarity}")
print(f"Average Expanded Num: {average_expanded_num}")
print(f"Average Planning Time Total: {average_planning_time_total}")
print(f"Average Cost Ratio: {average_cost_ratio}")
print(f"Average Current Cost: {average_current_cost}")

# Save daily detailed results and metrics to CSV
time_str = time.strftime('%Y%m%d', time.localtime())
details_filename = f'output_{name}_details_{time_str}.csv'
details_df = pd.DataFrame(test_results)
details_df.to_csv(details_filename, index=False)

metrics_filename = f'output_{name}_metrics_{time_str}.csv'
metrics_df.to_csv(metrics_filename, index=False)






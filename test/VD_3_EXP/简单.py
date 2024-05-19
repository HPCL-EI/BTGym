from btgym.algos.llm_client.llm_ask_tools import extract_llm_from_instr_goal, extract_llm_from_reflect, \
    convert_conditions, format_example
from tools import execute_algorithm, load_dataset, setup_default_env
from tools import find_from_small_act, load_dataset_and_cost
import time
import random
import numpy as np
import pandas as pd
import btgym
from btgym.utils import ROOT_PATH
from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process, act_format_records
from btgym.utils.tools import collect_action_nodes, extract_objects
from btgym.algos.llm_client.vector_database_env_goal import search_nearest_examples
from btgym.utils.read_dataset import read_dataset
from btgym.algos.llm_client.vector_database_env_goal import add_data_entry, write_metadata_to_txt, search_nearest_examples,add_to_database
import matplotlib.pyplot as plt
from generate_goals import random_generate_goals

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

def plot_results(metric_over_rounds, metric_name, name, sample_num):
    plt.figure()
    plt.plot(range(len(metric_over_rounds)), metric_over_rounds, marker='o')
    plt.xlabel('Round')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Over Rounds')
    plt.savefig(f'{name}_Det_round{round_num}_mr={max_round}_smpl={sample_num}_{metric_name.replace(" ", "_").lower()}.png')


def perform_test(env, chosen_goal, database_index_path, reflect_time=0, train=False, choose_database=True):
    cur_cond_set = env.agents[0].condition_set
    priority_act_ls, llm_key_pred, llm_key_obj, messages, distances, parsed_fail = \
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

        act_space = len(algo.actions)

        return success, np.mean(distances), priority_act_ls, key_predicates, key_objects, \
            act_num, error, time_limit_exceeded, current_cost, expanded_num, planning_time_total, act_space, parsed_fail
    else:
        success = False
        goal_set, priority_act_ls, key_predicates, key_objects, \
            act_num, error, time_limit_exceeded, current_cost, expanded_num, planning_time_total, act_space = \
            None, None, None, None, None, None, None, None, None, None, None

    # 搜索小动作空间得到一个解
    if not success and train:
        success, priority_act_ls, key_predicates, key_objects, \
            act_num, error, time_limit_exceeded, current_cost, expanded_num, planning_time_total, act_space = \
            find_from_small_act(chosen_goal)

    if not choose_database:
        distances = 0
    return success, np.mean(distances), priority_act_ls, key_predicates, key_objects, \
        act_num, error, time_limit_exceeded, current_cost, expanded_num, planning_time_total, act_space, parsed_fail


# Function to validate a single goal
def validate_goal(env, chosen_goal, n, database_index_path=None, round_num=None, database_num=None, reflect_time=0,
                  choose_database=True):
    print(f"test:{n}", chosen_goal)

    test_result = perform_test(env, chosen_goal, database_index_path, reflect_time=reflect_time,
                               choose_database=choose_database)
    success, avg_distance, priority_act_ls, key_predicates, key_objects, \
        act_num, error, time_limit_exceeded, current_cost, expanded_num, planning_time_total, \
        act_space, parsed_fail  = test_result

    fail = 0
    while not success:
        fail += 1
        RED = "\033[31m"
        RESET = "\033[0m"
        print(f"{RED}---- ID: {n}  Goal:{chosen_goal}    Fail Times: {fail} -----{RESET}")

        if fail > 6:
            # 跑全空间
            print(f"{RED}---- ID: {n}  Goal:{chosen_goal}  mode=\"big\" -----{RESET}")
            algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                                  priority_act_ls=priority_act_ls, key_predicates=key_predicates,
                                  key_objects=key_objects,
                                  selected_algorithm="opt", mode="big",
                                  llm_reflect=False, time_limit=10,
                                  heuristic_choice=0)
            goal_set = goal_transfer_str(' & '.join(chosen_goal))
            expanded_num, planning_time_total, cost, error, act_num, current_cost, record_act_ls = \
                execute_algorithm(algo, goal_set, cur_cond_set)
            time_limit_exceeded = algo.algo.time_limit_exceeded
            success = not error and not time_limit_exceeded
            act_space = len(algo.actions)
            break


        # 在这里添加例子，加入到
        # 如果这里面把例子中的pred和obj也加进去
        nearest_examples, distances = search_nearest_examples(database_index_path, llm, chosen_goal,
                                                              top_n=5 * 2 ** (fail - 1))
        ex_preds = set()
        ex_objs = set()
        for ex in nearest_examples:
            ex_preds |= set(ex['value']['Vital Action Predicates'].replace(" ", "").split(","))
            ex_objs |= set(ex['value']['Vital Objects'].replace(" ", "").split(","))
        key_predicates = list(set(key_predicates) | ex_preds)
        key_objects = list(set(key_objects) | ex_objs)

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
        act_space = len(algo.actions)


        if time_limit_exceeded:
            RED = "\033[31m"
            RESET = "\033[0m"
            print(f"{RED}- ID: {n}  Goal:{chosen_goal}  Time Out  -{RESET}")
        # 疑问：动作空间如何每次计算？没加一个要遍历所有动作？加了物体又怎么判别呢
        # 感觉可以加5个example、10个example、20个example....

        # VD 实验， 简单、依赖、长序列、混合？ 成功率和expanded？ 要扩张吗
        # 感觉体现向量数据库学习过程，可以不扩张，单纯看 四种数据集上的学习能力

    print(f"\033[92mtest:{n} {chosen_goal} {act_num}\033[0m")
    return {
        'round': round_num, 'id': n, 'goals': ' & '.join(chosen_goal), 'priority_act_ls': priority_act_ls,
        'key_predicates': key_predicates, 'key_objects': key_objects, 'act_num': act_num, 'error': error,
        'time_limit_exceeded': time_limit_exceeded, 'act_space': act_space, 'expanded_num': expanded_num, 'current_cost': current_cost,
        'planning_time_total': planning_time_total, 'average_distance': avg_distance, 'database_size': database_num
    }, success, avg_distance, fail,parsed_fail




# =========================  RHB ======================
name = "easy"
default_prompt_file = f"prompt_no_example.txt"
dataset = read_dataset(f"{name}_test_20.txt")
database_index_path = f"{ROOT_PATH}/../test/SCENES_EXP/DATABASE/0_goal_vectors.index"
from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction as RHB
env = btgym.make("VHT-PutMilkInFridge")
cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
cur_cond_set |= {f'IsClose({arg})' for arg in RHB.CAN_OPEN}
cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in RHB.HAS_SWITCH}
cur_cond_set |= {f'IsUnplugged({arg})' for arg in RHB.HAS_PLUG}
big_actions = collect_action_nodes(env.behavior_lib)



max_round = 1
sample_num = 2
vaild_num = 2
diffcult_type = "easy"

# Initialize accumulators and counters
test_results = []
test_success_count = 0
total_expanded_num = 0
total_planning_time_total = 0
total_cost_ratio = 0
total_current_cost = 0
total_fail_count = 0
total_act_space = 0
total_parsed_fail = 0
total_distance = 0


for round_num in range(0, 0 + max_round):

    metrics_over_rounds = {
        "Test Success Rate Once": [],
        "Average Distance": [],
        "Average Expanded Num": [],
        "Average Planning Time Total": [],
        "Average Current Cost": []
    }
    metrics_df = pd.DataFrame(columns=["Round", "Test Success Rate Once", "Average Distance", "Average Expanded Num",\
                                       "Average Planning Time Total", "Average Current Cost"])


    # 随机生成 20 个goals，放入 all_goals 集合
    if round_num != 0:
        round_goals = random_generate_goals(n=sample_num,diffcult_type=diffcult_type)
        # round_goals = random_medium_generate_goals(n=sample_num)
        # round_goals = random_hard_generate_goals(n=sample_num)
        successful_goals = []
        # 遍历所有的goal进行学习
        for n, chosen_goal in enumerate(round_goals):
            # chosen_goal=[abs, cde]
            chosen_goal = [s.strip() for s in chosen_goal.split('&')]
            print(f"\x1b[32m\n== Round: {round_num} ID: {n} {chosen_goal} \x1b[0m")
            train_result = perform_test(env, chosen_goal, database_index_path, choose_database=True,
                                        train=True)
            if train_result is None:
                print(f"\033[91mSkipping failed test result for goal: {chosen_goal}\033[0m")
                continue
            success, avg_distance, priority_act_ls, key_predicates, key_objects, \
                act_num, error, time_limit_exceeded, current_cost, expanded_num, planning_time_total, \
                act_space, parsed_fail = train_result
            # 如果成功就写入数据库
            if success:
                successful_goals.append({
                    "goal": " & ".join(chosen_goal),
                    "priority_act_ls": priority_act_ls,
                    "key_predicates": key_predicates,
                    "key_objects": key_objects,
                    "cost": current_cost
                })
        # 统一将所有成功的结果加入数据库
        for result in successful_goals:
            add_to_database(llm, env, result["goal"], result["priority_act_ls"], result["key_predicates"], result["key_objects"], database_index_path, result["cost"])

    # 更新一下向量数据库的数据数量
    # 读取存储的元数据
    metadata = np.load(database_index_path.replace(".index", "_metadata.npy"), allow_pickle=True)
    database_num = len(metadata)
    print(f"\033[91m Database Num: {database_num}\033[0m")

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
        result, success, avg_distance, fail, parsed_fail = validate_goal(env, d['Goals'], n, choose_database=True,
                                                                         database_index_path=database_index_path)
        test_results.append(result)
        # 计算一次成功率
        if success and fail == 0:
            test_success_count += 1

        total_distance += avg_distance
        total_expanded_num += result.get('expanded_num')
        total_planning_time_total += result.get('planning_time_total')
        total_fail_count += fail
        total_parsed_fail += parsed_fail
        total_act_space += result.get("act_space")

        total_cost_ratio += 0
        current_cost = 0
        total_current_cost += 0

    # Calculate metrics
    num_entries = len(vaild_dataset)
    success_rate = test_success_count / num_entries
    average_fail_count = total_fail_count / num_entries if num_entries else 0
    average_act_space = total_act_space / num_entries if num_entries else 0
    average_expanded_num = total_expanded_num / num_entries if num_entries else 0
    average_planning_time_total = total_planning_time_total / num_entries if num_entries else 0
    average_current_cost = total_current_cost / num_entries if num_entries else 0
    average_parsed_fail = total_parsed_fail / num_entries if num_entries else 0


    # Append metrics to dataframe
    round_metrics = pd.DataFrame([{
        "Test Success Rate": success_rate,
        "Average Fail Count": average_fail_count,
        "Average Act Space": average_act_space,
        "Average Expanded Num": average_expanded_num,
        "Average Planning Time Total": average_planning_time_total,
        "Average Current Cost": average_current_cost,
        "Average Parsed Fail": average_parsed_fail
    }])

    print(f"\033[92mTest Success Rate for Round {round_num}: {metrics_over_rounds['Test Success Rate'][-1]}\033[0m")
    print(
        f"\033[92mAverage Action Space Size for Round {round_num}: {metrics_over_rounds['Average Act Space'][-1]}\033[0m")
    print(
        f"\033[92mAverage Expanded Num for Round {round_num}: {metrics_over_rounds['Average Expanded Num'][-1]}\033[0m")
    print(
        f"\033[92mAverage Planning Time Total for Round {round_num}: {metrics_over_rounds['Average Planning Time Total'][-1]}\033[0m")
    print(
        f"\033[92mAverage Current Cost for Round {round_num}: {metrics_over_rounds['Average Current Cost'][-1]}\033[0m")
    print(f"\033[92mAverage Similarity for Round {round_num}: {metrics_over_rounds['Average Distance'][-1]}\033[0m")
    print(f"\033[92mDatabase Size for Round {round_num}: {database_num}\033[0m")


    if round_num % 5 == 0:
        # Save test results to CSV
        df = pd.DataFrame(test_results)
        time_str = time.strftime('%Y%m%d%H%M%S', time.localtime())
        df.to_csv(
            f'{name}_Det_round{round_num}_mr={max_round}_smpl={sample_num}.csv',
            index=False)

        # Save metrics results to CSV
        metrics_df.to_csv(
            f'{name}_Sum_round{round_num}_mr={max_round}_smpl={sample_num}.csv',
            index=False)

        # Plotting results
        for metric_name, metric_values in metrics_over_rounds.items():
            plot_results(metric_values, metric_name, name, sample_num)
        plt.show()


metrics_df = pd.concat([metrics_df, round_metrics], ignore_index=True)

# Output metrics
print(f"Test Success Rate: {success_rate}")
print(f"Average Parsed Fail: {average_parsed_fail}")
print(f"Average Fail Count: {average_fail_count}")
print(f"Average Act Space: {average_act_space}")
print(f"Average Expanded Num: {average_expanded_num}")
print(f"Average Planning Time Total: {average_planning_time_total}")
print(f"Average Current Cost: {average_current_cost}")

# Save daily detailed results and metrics to CSV
time_str = time.strftime('%Y%m%d', time.localtime())
details_filename = f'output_{name}_details_{time_str}.csv'
details_df = pd.DataFrame(test_results)
details_df.to_csv(details_filename, index=False)

metrics_filename = f'output_{name}_metrics_{time_str}.csv'
metrics_df.to_csv(metrics_filename, index=False)

# Set display options to ensure the entire DataFrame is printed
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# # Print the entire metrics dataframe
# print(metrics_df)

import pandas as pd
import matplotlib.pyplot as plt
import btgym
import time
from btgym import BehaviorTree
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction
from btgym.utils import ROOT_PATH
from btgym.utils.read_dataset import read_dataset
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process, act_format_records
from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
from btgym.algos.llm_client.llms.gpt4 import LLMGPT4
from btgym.algos.llm_client.llm_ask_tools import extract_llm_from_instr_goal
from tools import execute_algorithm, load_dataset, setup_default_env
from btgym.algos.llm_client.vector_database_env_goal import add_data_entry
import numpy as np
import random

# Set random seed
random_seed = 0
np.random.seed(random_seed)
random.seed(random_seed)

# Initialize the LLM
llm = LLMGPT3()
default_prompt_file = f"{ROOT_PATH}/algos/llm_client/prompt_VHT_just_goal_no_example.txt"
dataset = load_dataset(f"{ROOT_PATH}/../test/dataset/400data_processed_data.txt")
vaild_dataset = load_dataset(f"{ROOT_PATH}/../test/dataset/data1_env1_40_test_reflect.txt")

# Initialize variables
from ordered_set import OrderedSet

all_goals = OrderedSet()
for d in dataset:
    all_goals.update(d['Goals'])

group_id = 0
database_num = 5
test_results = []
test_success_rate_over_rounds = []
average_similarity_over_rounds = []
max_round = 30
sample_num = 10
vaild_num= 40


def add_to_database(env, goals, priority_act_ls, key_predicates, key_objects, database_index_path, cost):
    new_environment = str(env)
    new_goal = ' & '.join(goals)
    new_optimal_actions = ', '.join(priority_act_ls)
    new_vital_action_predicates = ', '.join(key_predicates)
    new_vital_objects = ', '.join(key_objects)
    add_data_entry(database_index_path, llm, new_environment, new_goal, new_optimal_actions,
                   new_vital_action_predicates, new_vital_objects, cost)
    print(f"\033[95mAdd the current data to the vector database\033[0m")


def perform_test(env, chosen_goal, database_index_path):
    cur_cond_set = setup_default_env()[1]
    priority_act_ls, llm_key_pred, llm_key_obj, messages, distances = \
        extract_llm_from_instr_goal(llm, default_prompt_file, 1, chosen_goal, verbose=False,
                                    choose_database=True, database_index_path=database_index_path)

    if priority_act_ls is None or llm_key_pred is None or llm_key_obj is None:
        return None  # Return None if any of the essential outputs are None

    _priority_act_ls, pred, obj = act_format_records(priority_act_ls)
    key_predicates = list(set(llm_key_pred + pred))
    key_objects = list(set(llm_key_obj + obj))
    algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                          priority_act_ls=priority_act_ls, key_predicates=key_predicates,
                          key_objects=key_objects,
                          selected_algorithm="opt", mode="small-predicate-objs",
                          llm_reflect=False, time_limit=10,
                          heuristic_choice=0)
    goal_set = goal_transfer_str(chosen_goal)
    expanded_num, planning_time_total, cost, error, act_num, current_cost, record_act_ls = \
        execute_algorithm(algo, goal_set, cur_cond_set)
    time_limit_exceeded = algo.algo.time_limit_exceeded
    success = not error and not time_limit_exceeded

    return success, np.mean(
        distances), _priority_act_ls, key_predicates, key_objects, cost, priority_act_ls, key_predicates, key_objects, \
        act_num, error, time_limit_exceeded, current_cost, expanded_num, planning_time_total


# Main Loop
for round_num in range(max_round):
    round_goals = random.sample(list(all_goals), sample_num)
    for goal in round_goals:
        all_goals.remove(goal)

    env, _ = setup_default_env()
    for n, chosen_goal in enumerate(round_goals):
        print(f"\x1b[32m\n== Round: {round_num} ID: {n} {chosen_goal} \x1b[0m")
        database_index_path = f"{ROOT_PATH}/../test/dataset/DATABASE/Group_{group_id}_env_goal_vectors.index"

        tarin_result = perform_test(env, chosen_goal, database_index_path)
        if tarin_result is None:
            print(f"\033[91mSkipping failed test result for goal: {chosen_goal}\033[0m")
            continue

        success, _, _priority_act_ls, key_predicates, key_objects, cost, _, _, _, _, _, _, _, _, _ = tarin_result
        if success:
            add_to_database(env, [chosen_goal], _priority_act_ls, key_predicates, key_objects, database_index_path,
                            cost)
            database_num += 1

    # Perform validation tests
    test_success_count = 0
    total_similarity = 0
    vaild_dataset_test = vaild_dataset[:vaild_num]
    for n, d in enumerate(vaild_dataset_test):
        chosen_goal = random.choice(d['Goals'])
        print(f"test:{n}",d['Goals'])

        test_result = perform_test(env, chosen_goal, database_index_path)
        if test_result is None:
            # Save the result with None marker
            test_results.append({
                'round': round_num,
                'id': n,
                'goals': chosen_goal,
                'priority_act_ls': None,
                'key_predicates': None,
                'key_objects': None,
                'act_num': None,
                'error': 'None',
                'time_limit_exceeded': 'None',
                'current_cost': None,
                'expanded_num': None,
                'planning_time_total': None,
                'average_distance': None,
                'database_size': database_num
            })
            print(f"\033[91mSkipping failed test result for goal: {chosen_goal}\033[0m")
            continue

        success, avg_similarity, _, \
            key_predicates, key_objects, _, priority_act_ls, key_predicates, key_objects, \
            act_num, error, time_limit_exceeded, current_cost, expanded_num, planning_time_total = \
                                test_result
        if success:
            test_success_count += 1
        total_similarity += avg_similarity

        # Append results for this validation goal
        test_results.append({
            'round': round_num,
            'id': n,
            'goals': chosen_goal,
            'priority_act_ls': priority_act_ls,
            'key_predicates': key_predicates,
            'key_objects': key_objects,
            'act_num': act_num,
            'error': error,
            'time_limit_exceeded': time_limit_exceeded,
            'current_cost': current_cost,
            'expanded_num': expanded_num,
            'planning_time_total': planning_time_total,
            'average_distance': avg_similarity,
            'database_size': database_num
        })

    test_success_rate = test_success_count / len(vaild_dataset_test)
    average_similarity = total_similarity / len(vaild_dataset_test)
    test_success_rate_over_rounds.append(test_success_rate)
    average_similarity_over_rounds.append(average_similarity)

    # Print the results in green
    print(f"\033[92mTest Success Rate for Round {round_num}: {test_success_rate}\033[0m")
    print(f"\033[92mAverage Similarity for Round {round_num}: {average_similarity}\033[0m")
    print(f"\033[92mDatabase Size for Round {round_num}: {database_num}\033[0m")

# Save test results to CSV
df = pd.DataFrame(test_results)
df.to_csv('EXP_2_database_test_results.csv', index=False)

# Plot the test success rate over rounds
plt.figure()
plt.plot(range(max_round), test_success_rate_over_rounds, marker='o')
plt.xlabel('Round')
plt.ylabel('Test Success Rate')
plt.title('Test Success Rate Over Rounds')
plt.savefig('EXP_2_test_success_rate_over_rounds.png')

# Plot the average similarity over rounds
plt.figure()
plt.plot(range(max_round), average_similarity_over_rounds, marker='o')
plt.xlabel('Round')
plt.ylabel('Average Similarity')
plt.title('Average Similarity Over Rounds')
plt.savefig('EXP_2_average_similarity_over_rounds.png')

plt.show()

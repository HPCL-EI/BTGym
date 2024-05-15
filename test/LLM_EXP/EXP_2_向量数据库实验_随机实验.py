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

random_seed = 0
np.random.seed(random_seed)
random.seed(random_seed)

# Initialize the LLM
llm = LLMGPT3()
default_prompt_file = f"{ROOT_PATH}/algos/llm_client/prompt_VHT_just_goal_no_example.txt"
dataset = load_dataset(f"{ROOT_PATH}/../test/dataset/400data_processed_data.txt")


# Initialize variables
from ordered_set import OrderedSet
all_goals = OrderedSet()
# Assuming dataset is defined
for id, d in enumerate(dataset):
    all_goals.update(d['Goals'])


start_time = time.time()
group_id = 0
# max_database_num = 100
database_num = 5

# To store the results
results = []

# Initialize similarity and success lists
similarity_over_rounds = []
success_rate_over_rounds = []
max_round=10
for round_num in range(max_round):
    round_goals = random.sample(list(all_goals), 40)
    for goal in round_goals:
        all_goals.remove(goal)

    round_distances = []
    round_success_count = 0

    for n, chosen_goal in enumerate(round_goals):
        print("Chosen goal:", chosen_goal)
        print(f"\x1b[32m\n== Round: {round_num} ID: {n} {chosen_goal} \x1b[0m")

        env, cur_cond_set = setup_default_env()
        database_index_path = f"{ROOT_PATH}/../test/dataset/DATABASE/Group_{group_id}_env_goal_vectors.index"

        priority_act_ls, llm_key_pred, llm_key_obj, messages, distances = \
            extract_llm_from_instr_goal(llm, default_prompt_file, 1, chosen_goal, verbose=False,
                                        choose_database=True, database_index_path=database_index_path)

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
        if success:
            round_success_count += 1

        # Append results for this goal
        results.append({
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
            'average_distance': np.mean(distances),
            'database_size': database_num  # This assumes the database size remains constant
        })

        round_distances.append(np.mean(distances))

        if success:
            new_environment = str(d['Environment'])
            new_goal = ' & '.join(d['Goals'])
            new_optimal_actions = ', '.join(_priority_act_ls)
            new_vital_action_predicates = ', '.join(key_predicates)
            new_vital_objects = ', '.join(key_objects)

            add_data_entry(database_index_path, llm, new_environment, new_goal, new_optimal_actions,
                           new_vital_action_predicates, new_vital_objects)
            print(f"\033[95mAdd the current data to the vector database\033[0m")

    # Calculate and store the average similarity and success rate for this round
    avg_similarity = np.mean(round_distances)
    success_rate = round_success_count / len(round_goals)

    similarity_over_rounds.append(avg_similarity)
    success_rate_over_rounds.append(success_rate)

# Convert the results to a DataFrame and save to CSV
df = pd.DataFrame(results)
df.to_csv('EXP_2_database_results.csv', index=False)

# Plot the similarity over rounds
plt.figure()
plt.plot(range(max_round), similarity_over_rounds, marker='o')
plt.xlabel('Round')
plt.ylabel('Average Similarity')
plt.title('Average Similarity Over Rounds')
plt.savefig('similarity_over_rounds.png')

# Plot the success rate over rounds
plt.figure()
plt.plot(range(max_round), success_rate_over_rounds, marker='o')
plt.xlabel('Round')
plt.ylabel('Success Rate')
plt.title('Success Rate Over Rounds')
plt.savefig('success_rate_over_rounds.png')

plt.show()

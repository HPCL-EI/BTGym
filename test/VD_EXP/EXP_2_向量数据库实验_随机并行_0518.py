import pandas as pd
import matplotlib.pyplot as plt
import btgym
import time
from btgym import BehaviorTree
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from btgym.utils import ROOT_PATH
from btgym.utils.read_dataset import read_dataset
from btgym.algos.llm_client.tools import goal_transfer_str, act_format_records
from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
from btgym.algos.llm_client.llm_ask_tools import extract_llm_from_instr_goal, extract_llm_from_reflect, convert_conditions, format_example
from tools import execute_algorithm, load_dataset, setup_default_env
from btgym.algos.llm_client.vector_database_env_goal import add_data_entry, write_metadata_to_txt, search_nearest_examples,add_to_database
import numpy as np
import random
from ordered_set import OrderedSet
import concurrent.futures
from generate_goals import random_generate_goals
from tools import find_from_small_act

# Set random seed
random_seed = 0
np.random.seed(random_seed)
random.seed(random_seed)

# Initialize the LLM
llm = LLMGPT3()

ACT_PREDICATES = {"Walk", "RightGrab", "LeftGrab", "RightPut", "LeftPut", "RightPutIn", "LeftPutIn", \
                  "Open", "Close", "SwitchOn", "SwitchOff", "Wipe", "PlugIn", "PlugOut", "Cut", "Wash"}
TOOLS = {'kitchenknife', 'faucet', 'rag'}

def plot_results(metric_over_rounds, metric_name, group_id, name, sample_num, reflect_num):
    plt.figure()
    plt.plot(range(len(metric_over_rounds)), metric_over_rounds, marker='o')
    plt.xlabel('Round')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Over Rounds')
    plt.savefig(f'output_ref={reflect_time}/EXP_2_{metric_name.replace(" ", "_").lower()}rf={reflect_time}_round{round_num}_G{group_id}_{name}_mr={max_round}_smpl={sample_num}_paral.png')

def convert_set_to_str(string_set):
    return ", ".join(f'\"{s}\"' for s in string_set)

def reflect_on_errors(llm, messages, d, env, cur_cond_set, goal_set, priority_act_ls, key_predicates, key_objects):
    # 查询还有哪些动作谓词没有用到
    not_use_pred = ACT_PREDICATES - set(key_predicates)
    not_use_pred_str = convert_set_to_str(not_use_pred)
    not_use_obj = TOOLS - set(key_objects)
    if not_use_obj != set():
        not_use_obj_str = ", and the tools/objects you have not used are: " + convert_set_to_str(not_use_obj)+""
    else:
        not_use_obj_str = ""

    reflect_prompt = (
        "The list of actions, predicates, and objects you provided is insufficient to accomplish the specified goals: \"{goals}\". "
        "Specifically, these only allow for the completion of the \"{have_finished}\", while failing to address the \"{not_finished}\".\n"

        "Note that you have not used the following Action Predicates: {not_use_pred_str}{not_use_obj_str}."
        "In regards to the unfinished goals {not_finished}, check if these unused action predicates and objects are important and helpful for completing the goals. Please try to include any vital missing action predicates and objects.\n"

        "For the unfinished goals \"{not_finished}\", you can refer to the example below.\n"

        "Please re-analyze the specified goal to identify the optimal actions, essential action predicates, and key objects necessary for achieving the goals. "
        "Use the same format as previously used, beginning with 'Optimal Actions:', 'Vital Action Predicates:', and 'Vital Objects:' respectively. Do not provide any additional explanations."
        
        "[Example]"
    )

    not_finished = set()
    for _g in d["Goals"]:
        #  _g = 'IsCut_pear'
        algo_tmp = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                                  priority_act_ls=priority_act_ls, key_predicates=key_predicates,
                                  key_objects=key_objects,
                                  selected_algorithm="opt", mode="small-predicate-objs",
                                  llm_reflect=False, time_limit=3,
                                  heuristic_choice=0)
        gset = goal_transfer_str(_g)  # gset = [{'IsCut(pear)'}]
        algo_tmp.process(gset)
        state_g = cur_cond_set
        g = gset[0]  # g='IsCut(pear)'
        error_g, state_g, _, _, _ = algo_tmp.execute_bt(g, state_g, verbose=False)
        if error_g:
            not_finished |= g

    have_finished = goal_set[0] - not_finished
    # goal_set[0] = {[{'IsClean(nightstand)','IsCut(pear)','IsOpen(window)','IsPlugged(wallphone)','IsSwitchedOn(wallphone)'}]}
    have_finished_str = convert_conditions(have_finished)
    not_finished_str = convert_conditions(not_finished)

    reflect_prompt = reflect_prompt.format(goals=' & '.join(d['Goals']), have_finished=have_finished_str,
                                           not_finished=not_finished_str,
                                           not_use_pred_str=not_use_pred_str, not_use_obj_str=not_use_obj_str)
    messages.append({"role": "user", "content": reflect_prompt})

    # ================ 再加5个例子 ===========
    reflect_goals = " & ".join(list(not_finished))
    nearest_examples, distances = search_nearest_examples(database_index_path, llm, reflect_goals, top_n=5)
    # 使用自定义的格式函数将检索到的示例格式化为目标样式
    example_texts = '\n'.join([format_example(ex) for ex in nearest_examples])
    example_texts = "[Examples]\n" + example_texts

    # 输出最近的所有goal
    nearest_goals = [ex['value']['Goals'] for ex in nearest_examples]
    print("Reflect: All Goals from nearest examples:")
    for g in nearest_goals:
        print(f"\033[93m{g}\033[0m")  # 打印黄色 print(goal)
    example_marker = "[Examples]"
    if example_marker in reflect_prompt:
        reflect_prompt = reflect_prompt.replace(example_marker, example_texts)
    else:
        reflect_prompt = f"{reflect_prompt}\n{example_texts}"
    # ================ 再加5个例子 ===========


    print("reflect_prompt:", reflect_prompt)

    return extract_llm_from_reflect(llm, messages)

def perform_test(env, chosen_goal, database_index_path, reflect_time=0,train=False,choose_database=True):
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
        goal_set,key_predicates,key_objects=None,None,None

    # 写反馈
    fail_time = 0
    while not success and fail_time < reflect_time:
        fail_time += 1
        print(f"大模型重推荐......fail_time={fail_time}")
        if priority_act_ls != None:
            priority_act_ls_new, llm_key_pred_new, llm_key_obj_new, messages = reflect_on_errors(llm, messages, d, env,
                                                                                                 cur_cond_set, goal_set,
                                                                                                 priority_act_ls,
                                                                                                 key_predicates,
                                                                                                 key_objects)
        else:
            priority_act_ls_new, llm_key_pred_new, llm_key_obj_new, messages, distances = \
                extract_llm_from_instr_goal(llm, default_prompt_file, 1, chosen_goal, verbose=False,
                                            choose_database=choose_database, database_index_path=database_index_path)
        if priority_act_ls == None or priority_act_ls_new==None:
            continue

        priority_act_ls = list(OrderedSet(priority_act_ls + priority_act_ls_new))
        llm_key_pred = list(OrderedSet(llm_key_pred + llm_key_pred_new))
        llm_key_obj = list(OrderedSet(llm_key_obj + llm_key_obj_new))

        _, pred, obj = act_format_records(priority_act_ls)
        key_predicates = list(set(llm_key_pred + pred))
        key_objects = list(set(llm_key_obj + obj))

        algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                              priority_act_ls=priority_act_ls, key_predicates=key_predicates,
                              key_objects=key_objects,
                              selected_algorithm="opt", mode="small-predicate-objs",
                              llm_reflect=False, time_limit=10,
                              heuristic_choice=0)
        expanded_num, planning_time_total, cost, error, act_num, current_cost, record_act_ls = \
            execute_algorithm(algo, goal_set, cur_cond_set)
        time_limit_exceeded = algo.algo.time_limit_exceeded
        success = not error and not time_limit_exceeded

        if success :
            print(
                f"\033[92mReflect: Success After reflect!\033[0m")


    if not success and train:
        # 搜索小动作空间得到一个解
        success,  _priority_act_ls, key_predicates, key_objects, cost, priority_act_ls, key_predicates, key_objects, \
            act_num, error, time_limit_exceeded, current_cost, expanded_num, planning_time_total = find_from_small_act(chosen_goal)


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
def validate_goal(env, chosen_goal, database_index_path, round_num, n, database_num, reflect_time=0,choose_database=True):
    print(f"test:{n}", chosen_goal)

    test_result = perform_test(env, chosen_goal, database_index_path, reflect_time=reflect_time,choose_database=choose_database)
    success, avg_similarity, _, key_predicates, key_objects, _, priority_act_ls, key_predicates, key_objects, \
        act_num, error, time_limit_exceeded, current_cost, expanded_num, planning_time_total = test_result

    if not success:
        return {
            'round': round_num, 'id': n, 'goals': ' & '.join(chosen_goal), 'priority_act_ls': None, 'key_predicates': None, 'key_objects': None, 'act_num': None, 'error': 'None', 'time_limit_exceeded': 'None', 'current_cost': None, 'expanded_num': None, 'planning_time_total': None, 'average_distance': None, 'database_size': database_num
        }, False, avg_similarity

    print(f"\033[92mtest:{n} {chosen_goal} {act_num}\033[0m")
    return {
        'round': round_num, 'id': n, 'goals': ' & '.join(chosen_goal), 'priority_act_ls': priority_act_ls, 'key_predicates': key_predicates, 'key_objects': key_objects, 'act_num': act_num, 'error': error, 'time_limit_exceeded': time_limit_exceeded, 'current_cost': current_cost, 'expanded_num': expanded_num, 'planning_time_total': planning_time_total, 'average_distance': avg_similarity, 'database_size': database_num
    }, success, avg_similarity



default_prompt_file = f"{ROOT_PATH}/algos/llm_client/prompt_VHT_just_goal_no_example.txt"
# default_prompt_file = f"{ROOT_PATH}/algos/llm_client/prompt_VHT_just_goal.txt"
train_dataset = load_dataset(f"{ROOT_PATH}/../test/dataset/400data_processed_data.txt")
all_goals = []
for id, d in enumerate(train_dataset):
    # all_goals.append(' & '.join(d['Goals']))
    all_goals.append(d['Goals'])

use_random = True

# vaild_dataset = load_dataset(f"test_data_40_0518_2.txt")
# vaild_dataset = load_dataset(f"test_data_40_0517_single.txt")
vaild_dataset = load_dataset(f"test_data_20_0518_3.txt")
# vaild_dataset = load_dataset(f"{ROOT_PATH}/../test/dataset/DATA_BT_100_ori_yz_revby_cys.txt")
# vaild_dataset = load_dataset(f"{ROOT_PATH}/../test/dataset/data1_env1_40_test_reflect.txt")

group_id = '0'
database_num = 5
env, _ = setup_default_env()
database_index_path = f"{ROOT_PATH}/../test/VD_EXP/DATABASE/Group{group_id}_env_goal_vectors.index"
database_output_path = f"{ROOT_PATH}/../test/VD_EXP/DATABASE/Group{group_id}_env_goal_vectors.txt"

# max_round = 10 +1
# sample_num = 10
# vaild_num = 20 #40

max_round = 2 +1
sample_num = 1
vaild_num = 2 #40

for reflect_time in [0,1]:



    test_results = []
    metrics_over_rounds = {
        "Test Success Rate": [],
        "Average Distance": [],
        "Average Expanded Num": [],
        "Average Planning Time Total": [],
        "Average Current Cost": []
    }
    # Initialize an empty DataFrame to store the metrics for each round
    metrics_df = pd.DataFrame(columns=["Round", "Test Success Rate", "Average Distance", "Average Expanded Num", "Average Planning Time Total", "Average Current Cost"])


    # Main Loop
    for round_num in range(max_round):
        successful_goals = []
        # 训练过程
        if round_num != 0:
            if use_random:
                round_goals = random_generate_goals(n=sample_num)  # 随机生成 20 个goals，放入 all_goals 集合
            else:
                round_goals = random.sample(all_goals, sample_num)  # 每次读取数据里的data
                for goal in round_goals:
                    all_goals.remove(goal)
            # 遍历所有的goal进行学习
            for n, chosen_goal in enumerate(round_goals):
                # chosen_goal=[abs, cde]
                chosen_goal = [s.strip() for s in chosen_goal.split('&')]
                print(f"\x1b[32m\n== Round: {round_num} ID: {n} {chosen_goal} \x1b[0m")

                tarin_result = perform_test(env, chosen_goal, database_index_path, reflect_time=reflect_time,\
                                            train=True)
                if tarin_result is None:
                    print(f"\033[91mSkipping failed test result for goal: {chosen_goal}\033[0m")
                    continue
                # 如果成功就写入数据库
                success, _, _priority_act_ls, key_predicates, key_objects, cost, _, _, _, _, _, _, _, _, _ = tarin_result
                # if success:
                #     add_to_database(llm, env, " & ".join(chosen_goal), _priority_act_ls, key_predicates, key_objects,
                #                     database_index_path,
                #                     cost)
                if success:
                    successful_goals.append({
                        "goal": " & ".join(chosen_goal),
                        "priority_act_ls": _priority_act_ls,
                        "key_predicates": key_predicates,
                        "key_objects": key_objects,
                        "cost": cost
                    })

            # 统一将所有成功的结果加入数据库
            for result in successful_goals:
                add_to_database(llm, env, result["goal"], result["priority_act_ls"], result["key_predicates"], result["key_objects"], database_index_path, result["cost"])


        # 更新一下向量数据库的数据数量
        # 读取存储的元数据
        metadata = np.load(database_index_path.replace(".index", "_metadata.npy"), allow_pickle=True)
        database_num = len(metadata)
        print(f"\033[91m Database Num: {database_num}\033[0m")

        # Perform validation tests in parallel
        test_success_count = 0
        total_similarity = 0
        total_expanded_num = 0
        total_planning_time_total = 0
        total_current_cost = 0
        vaild_dataset_test = vaild_dataset[:vaild_num]
        # vaild_dataset_test = vaild_dataset[9:9+vaild_num]

        # ========================= 并行 ========================
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(validate_goal, env, d['Goals'], database_index_path, round_num, n, database_num,
                                       reflect_time=reflect_time,choose_database=True) \
                       for n, d in enumerate(vaild_dataset_test)]
            for future in concurrent.futures.as_completed(futures):
                result, success, avg_similarity = future.result()
                test_results.append(result)
                if success:
                    test_success_count += 1
                total_similarity += avg_similarity
                total_expanded_num += result['expanded_num'] if result['expanded_num'] is not None else 300
                total_planning_time_total += result['planning_time_total'] if result[
                                                                                  'planning_time_total'] is not None else 3
                total_current_cost += result['current_cost'] if result['current_cost'] is not None else 2000
        # ========================= 并行 ========================
        # ========================= 串行========================
        # for n, d in enumerate(vaild_dataset_test):
        #     result, success, avg_similarity = validate_goal(env, d['Goals'], database_index_path, round_num, n, database_num,
        #                                reflect_time=reflect_time)
        #     test_results.append(result)
        #     if success:
        #         test_success_count += 1
        #     total_similarity += avg_similarity
        #     total_expanded_num += result['expanded_num'] if result['expanded_num'] is not None else 300
        #     total_planning_time_total += result['planning_time_total'] if result[
        #                                                                       'planning_time_total'] is not None else 3
        #     total_current_cost += result['current_cost'] if result['current_cost'] is not None else 2000
        # ========================= 串行========================


        metrics_over_rounds["Test Success Rate"] = np.append(metrics_over_rounds["Test Success Rate"], test_success_count / len(vaild_dataset_test))
        metrics_over_rounds["Average Distance"] = np.append(metrics_over_rounds["Average Distance"], total_similarity / len(vaild_dataset_test))
        metrics_over_rounds["Average Expanded Num"] = np.append(metrics_over_rounds["Average Expanded Num"], total_expanded_num / len(vaild_dataset_test))
        metrics_over_rounds["Average Planning Time Total"] = np.append(metrics_over_rounds["Average Planning Time Total"], total_planning_time_total / len(vaild_dataset_test))
        metrics_over_rounds["Average Current Cost"] = np.append(metrics_over_rounds["Average Current Cost"], total_current_cost / len(vaild_dataset_test))

        # Append the metrics of the current round to the DataFrame
        round_metrics = pd.DataFrame([{
            "Round": round_num,
            "Test Success Rate": metrics_over_rounds["Test Success Rate"][-1],
            "Average Distance": metrics_over_rounds["Average Distance"][-1],
            "Average Expanded Num": metrics_over_rounds["Average Expanded Num"][-1],
            "Average Planning Time Total": metrics_over_rounds["Average Planning Time Total"][-1],
            "Average Current Cost": metrics_over_rounds["Average Current Cost"][-1]
        }])
        metrics_df = pd.concat([metrics_df, round_metrics], ignore_index=True)

        print(f"\033[92mTest Success Rate for Round {round_num}: {metrics_over_rounds['Test Success Rate'][-1]}\033[0m")
        print(f"\033[92mAverage Similarity for Round {round_num}: {metrics_over_rounds['Average Distance'][-1]}\033[0m")
        print(f"\033[92mAverage Expanded Num for Round {round_num}: {metrics_over_rounds['Average Expanded Num'][-1]}\033[0m")
        print(f"\033[92mAverage Planning Time Total for Round {round_num}: {metrics_over_rounds['Average Planning Time Total'][-1]}\033[0m")
        print(f"\033[92mAverage Current Cost for Round {round_num}: {metrics_over_rounds['Average Current Cost'][-1]}\033[0m")
        print(f"\033[92mDatabase Size for Round {round_num}: {database_num}\033[0m")

        # 结束以后将向量数据库保存为 txt 文件
        # 将向量数据库里的所有数据写入 txt
        # Save the current round's database to a file with the round number in the filename
        write_metadata_to_txt(database_index_path,
                              f"{ROOT_PATH}/../test/VD_EXP/output_ref={reflect_time}/DB_rf={reflect_time}_round{round_num}_G{group_id}.txt")
        # Save the current round's vector database index file
        round_index_path = f"{ROOT_PATH}/../test/VD_EXP/output_ref={reflect_time}/DB_rf={reflect_time}_round{round_num}_G{group_id}.index"
        with open(database_index_path, 'rb') as f_src:
            with open(round_index_path, 'wb') as f_dst:
                f_dst.write(f_src.read())

        if round_num%5==0:
            # Save test results to CSV
            df = pd.DataFrame(test_results)
            if use_random:
                name = "Random"
            else:
                name = "400data"
            time_str = time.strftime('%Y%m%d%H%M%S', time.localtime())
            df.to_csv(f'output_ref={reflect_time}/EXP_2_DT_Det_rf={reflect_time}_round{round_num}_G{group_id}_{name}_mr={max_round}_smpl={sample_num}_paral.csv', index=False)

            # Save metrics results to CSV
            metrics_df.to_csv(f'output_ref={reflect_time}/EXP_2_DT_Sum_rf={reflect_time}_round{round_num}_G{group_id}_{name}_mr={max_round}_smpl={sample_num}_paral.csv', index=False)

            # Plotting results
            for metric_name, metric_values in metrics_over_rounds.items():
                plot_results(metric_values, metric_name, group_id, name, sample_num, reflect_time)
            plt.show()
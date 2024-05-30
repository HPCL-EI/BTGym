import pandas as pd
import btgym
import time
from ordered_set import OrderedSet
from btgym import BehaviorTree
from btgym.algos.bt_autogen.main_interface import BTExpInterface, collect_conditions
from btgym.envs.RobotHow.exec_lib._base.RHAction import VHTAction
from btgym.utils import ROOT_PATH
from btgym.utils.read_dataset import read_dataset
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process, act_format_records
from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
from btgym.algos.llm_client.llms.gpt4 import LLMGPT4
from btgym.algos.llm_client.llm_ask_tools import extract_llm_from_instr_goal, convert_conditions, \
    extract_llm_from_reflect,format_example
from btgym.algos.llm_client.vector_database_env_goal import add_data_entry, write_metadata_to_txt, search_nearest_examples,add_to_database


def load_result_csv(file_name):
    results_df = pd.read_csv(file_name)
    results = results_df.to_dict(orient='records')
    print(f"Loaded {len(results)} results from '{file_name}'")
    return results


def load_dataset(data_path):
    data = read_dataset(data_path)
    print(f"导入 {len(data)} 条数据")
    return data


def setup_env():
    env = btgym.make("VHT-PutMilkInFridge")
    cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
    cur_cond_set |= {f'IsClose({arg})' for arg in VHTAction.CAN_OPEN}
    cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHTAction.HAS_SWITCH}
    cur_cond_set |= {f'IsUnplugged({arg})' for arg in VHTAction.HAS_PLUG}
    return env, cur_cond_set


def execute_algorithm(algo, goal_set, cur_cond_set):
    start_time = time.time()
    algo.process(goal_set)
    end_time = time.time()
    planning_time_total = end_time - start_time

    ptml_string, cost, expanded_num = algo.post_process()
    error, state, act_num, current_cost, record_act_ls = algo.execute_bt(goal_set[0], cur_cond_set, verbose=False)

    return expanded_num, planning_time_total, cost, error, act_num, current_cost, record_act_ls


def recommend_actions(llm, default_prompt_file, instruction, goals):
    return extract_llm_from_instr_goal(llm, default_prompt_file, instruction, goals, verbose=False)


# def reflect_on_errors(llm, messages, d, env, cur_cond_set, goal_set, priority_act_ls, key_predicates, key_objects):
#     reflect_prompt = (
#         "The list of actions, predicates, and objects you provided is insufficient to accomplish the specified goals: {goals}. "
#         "Specifically, these only allow for the completion of the {have_finished}, while failing to address the {not_finished}.\n"
#
#         "1. In regards to the {not_finished}, several critical dependencies and tools have been overlooked. "
#         "This includes the need to plug in an electrical appliance before using it, and the need to open a container before placing items inside.\n "
#         "2. Additionally, for the {not_finished}, it appears that essential tools like 'rag', 'faucet', 'kitchenknife' and steps have also been neglected. "
#         "The requirements include using a rag for wiping, going to the faucet and turning it on for washing, wiping and washing to clean an object, and using a kitchen knife for cutting.\n"
#         "3. Moreover, for the {not_finished}, some actions require coordination between the hands that was not managed correctly. "
#         "For example, handling different objects simultaneously with both hands when necessary.\n"
#
#         "Please re-analyze the specified goal to identify the optimal actions, essential action predicates, and key objects necessary for achieving the goals. "
#         "Use the same format as previously used, beginning with 'Optimal Actions:', 'Vital Action Predicates:', and 'Vital Objects:' respectively."
#
#         # 参考例子中 Vital Action Predicates  Vital Objects
#     )
#
#     not_finished = set()
#     for _g in d["Goals"]:
#         #  _g = 'IsCut_pear'
#         algo_tmp = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
#                                   priority_act_ls=priority_act_ls, key_predicates=key_predicates,
#                                   key_objects=key_objects,
#                                   selected_algorithm="opt", mode="small-predicate-objs",
#                                   llm_reflect=False, time_limit=3,
#                                   heuristic_choice=0)
#         gset = goal_transfer_str(_g)  # gset = [{'IsCut(pear)'}]
#         algo_tmp.process(gset)
#         state_g = cur_cond_set
#         g = gset[0]  # g='IsCut(pear)'
#         error_g, state_g, _, _, _ = algo_tmp.execute_bt(g, state_g, verbose=False)
#         if error_g:
#             not_finished |= g
#
#     have_finished = goal_set[0] - not_finished
#     # goal_set[0] = {[{'IsClean(nightstand)','IsCut(pear)','IsOpen(window)','IsPlugged(wallphone)','IsSwitchedOn(wallphone)'}]}
#     have_finished_str = convert_conditions(have_finished)
#     not_finished_str = convert_conditions(not_finished)
#
#     reflect_prompt = reflect_prompt.format(goals=d['Goals'], have_finished=have_finished_str,
#                                            not_finished=not_finished_str)
#     messages.append({"role": "user", "content": reflect_prompt})
#     print("reflect_prompt:", reflect_prompt)
#
#     return extract_llm_from_reflect(llm, messages)

ACT_PREDICATES = {"Walk", "RightGrab", "LeftGrab", "RightPut", "LeftPut", "RightPutIn", "LeftPutIn", \
                  "Open", "Close", "SwitchOn", "SwitchOff", "Wipe", "PlugIn", "PlugOut", "Cut", "Wash"}
TOOLS = {'kitchenknife', 'faucet', 'rag'}
def convert_set_to_str(string_set):
    return ", ".join(f'\"{s}\"' for s in string_set)

def reflect_on_errors(llm, messages, d, env, cur_cond_set, goal_set, priority_act_ls, key_predicates, key_objects):
    # 查询还有哪些动作谓词没有用到
    not_use_pred = ACT_PREDICATES - set(key_predicates)
    not_use_pred_str = convert_set_to_str(not_use_pred)
    not_use_obj = TOOLS - set(key_objects)
    if not_use_obj != set():
        not_use_obj_str = ", and the tools/objects you have not used are: " + convert_set_to_str(not_use_obj) + ""
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

    reflect_prompt = reflect_prompt.format(goals=d['Goals'], have_finished=have_finished_str,
                                           not_finished=not_finished_str,
                                           not_use_pred_str=not_use_pred_str, not_use_obj_str=not_use_obj_str)
    messages.append({"role": "user", "content": reflect_prompt})

    # ================ 再加5个例子 ===========
    reflect_goals = " & ".join(list(not_finished))
    database_index_path = f"{ROOT_PATH}/../test/VD_EXP/DATABASE/Group400_env_goal_vectors.index"
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

# 主函数
llm = LLMGPT3()
default_prompt_file = f"{ROOT_PATH}\\algos\\llm_client\\prompt_VHT_just_goal.txt"
dataset = load_dataset(f"{ROOT_PATH}/../test/dataset/data1_env1_40_test_reflect.txt")

results = []
start_time = time.time()

for id, d in enumerate(dataset[20:30]):
    # print("\x1b[32m\n== ID:", id, "  ", d['Goals'],"\x1b[0m")
    print("\n== ID:", id, "  ", d['Goals'])
    instruction = d['Instruction']
    goals = d['Goals']
    d['Optimal Actions'] = act_str_process(d['Optimal Actions'], already_split=True)

    result_entry = {
        'id': id,
        'Instruction': instruction,
        'Goals': goals,
        'Optimal Actions': d['Optimal Actions'],
        'Vital Action Predicates': d['Vital Action Predicates'],
        'Vital Objects': d['Vital Objects']
    }

    env, cur_cond_set = setup_env()

    # Initial recommendation from LLM
    priority_act_ls, llm_key_pred, llm_key_obj, messages,_ = recommend_actions(llm, default_prompt_file, instruction,
                                                                             goals)
    print("Rec Act:", priority_act_ls)
    print("Rec Pred", llm_key_pred)
    print("Rec Obj:", llm_key_obj)

    _, pred, obj = act_format_records(priority_act_ls)
    key_predicates = list(set(llm_key_pred + pred))
    key_objects = list(set(llm_key_obj + obj))

    algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                          priority_act_ls=priority_act_ls, key_predicates=key_predicates,
                          key_objects=key_objects,
                          selected_algorithm="opt", mode="small-predicate-objs",
                          llm_reflect=False, time_limit=10,
                          heuristic_choice=0)

    goal_set = goal_transfer_str(' & '.join(goals))
    expanded_num, planning_time_total, cost, error, act_num, current_cost, record_act_ls = execute_algorithm(algo,
                                                                                                             goal_set,
                                                                                                             cur_cond_set)
    print(
        f"Expanded Conditions: {expanded_num}, planning_time_total: {planning_time_total}, cost_total: {cost}, action_steps: {act_num}, current_cost: {current_cost}")

    fail_time = 0
    while error and fail_time < 3:
        fail_time += 1
        print(f"大模型重推荐......fail_time={fail_time}")

        priority_act_ls_new, llm_key_pred_new, llm_key_obj_new, messages = reflect_on_errors(llm, messages, d, env,
                                                                                             cur_cond_set, goal_set,
                                                                                             priority_act_ls,
                                                                                             key_predicates,
                                                                                             key_objects)
        priority_act_ls = list(OrderedSet(priority_act_ls + priority_act_ls_new))
        llm_key_pred = list(OrderedSet(llm_key_pred + llm_key_pred_new))
        llm_key_obj = list(OrderedSet(llm_key_obj + llm_key_obj_new))

        print("Rec Act:", priority_act_ls)
        print("Rec Pred", llm_key_pred)
        print("Rec Obj:", llm_key_obj)

        _, pred, obj = act_format_records(priority_act_ls)
        key_predicates = list(set(llm_key_pred + pred))
        key_objects = list(set(llm_key_obj + obj))

        algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                              priority_act_ls=priority_act_ls, key_predicates=key_predicates,
                              key_objects=key_objects,
                              selected_algorithm="opt", mode="small-predicate-objs",
                              llm_reflect=False, time_limit=10,
                              heuristic_choice=0)

        expanded_num, planning_time_total, cost, error, act_num, current_cost, record_act_ls = execute_algorithm(
            algo, goal_set, cur_cond_set)
        print(
            f"Expanded Conditions: {expanded_num}, planning_time_total: {planning_time_total}, cost_total: {cost}, action_steps: {act_num - 1}, current_cost: {current_cost}")

        if not error :
            print(
                f"\033[92m Success After reflect!\033[0m")

    result_entry.update({
        'reflect': fail_time,
        'Timeout': 1 if algo.algo.time_limit_exceeded else 0,
        'err': 1 if error else 0,
        'exp': expanded_num,
        'time': planning_time_total,
        'cost': cost,
        'act': record_act_ls
    })

    results.append(result_entry)

df = pd.DataFrame(results)
df.to_csv(f"LLM_40_reflect.csv", index=False)
print("Results have been saved to LLM_40_reflect.csv")

end_time = time.time()
print("Total time:", end_time - start_time)


# Calculate success rates
total_tests = len(results)
successful_tests = sum(1 for result in results if result['err'] == 0)
success_after_reflect = sum(1 for result in results if result['err'] == 0 and result['reflect'] > 0)

total_success_rate = successful_tests / total_tests
success_rate_after_reflect = success_after_reflect / total_tests

print(f"Total success rate: {total_success_rate * 100:.2f}%")
print(f"Success rate after reflection: {success_rate_after_reflect * 100:.2f}%")
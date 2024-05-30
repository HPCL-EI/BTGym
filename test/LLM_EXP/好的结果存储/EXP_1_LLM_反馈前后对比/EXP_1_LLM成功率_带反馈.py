import pandas as pd
import btgym
import time
from ordered_set import OrderedSet
from btgym import BehaviorTree
from btgym.algos.bt_autogen.main_interface import BTExpInterface,collect_conditions
from btgym.envs.RobotHow.exec_lib._base.RHAction import VHTAction
from btgym.utils import ROOT_PATH
from btgym.utils.read_dataset import read_dataset
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process, act_format_records
from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
from btgym.algos.llm_client.llms.gpt4 import LLMGPT4
from btgym.algos.llm_client.llm_ask_tools import extract_llm_from_instr_goal,convert_conditions,extract_llm_from_reflect
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
llm=LLMGPT3()
default_prompt_file = f"{ROOT_PATH}\\algos\\llm_client\\prompt_VHT_just_goal.txt"

# 导入数据集 真实值
dataset1 = load_dataset(f"{ROOT_PATH}/../test/dataset/data1_env1_40_test_reflect.txt")

results = []
start_time = time.time()
for id, d in enumerate(dataset1): # 5可以
    print("\n== ID:", id, "  ", d['Goals'])

    instruction = d['Instruction']
    goals = d['Goals']
    d['Optimal Actions'] = act_str_process(d['Optimal Actions'], already_split=True)


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

    # 大模型推荐的结果
    priority_act_ls, llm_key_pred, llm_key_obj, messages = \
        extract_llm_from_instr_goal(llm, default_prompt_file, instruction, goals, verbose=False)
    # 增加板块，解析错误，直接重来，3次以上认定为错误

    print("Rec Act:", priority_act_ls)
    print("Rec Pred", llm_key_pred)
    print("Rec Obj:", llm_key_obj)

    # key_predicates 和 key_objects 要将推荐的 priority_act_ls 补充进来
    _, pred, obj = act_format_records(priority_act_ls)
    key_predicates = list(set(llm_key_pred + pred))
    key_objects = list(set(llm_key_obj + obj))

    algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set, \
                          priority_act_ls=priority_act_ls, key_predicates=key_predicates,
                          key_objects=key_objects, \
                          selected_algorithm="opt", mode="small-predicate-objs", \
                          llm_reflect=False, time_limit=10,
                          heuristic_choice=0)

    start_time = time.time()
    goal_set = goal_transfer_str(' & '.join(d["Goals"]))
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

    # 失败就要重试，最多三次
    fail_time=0
    while error and fail_time<=3:

        print(f"大模型重推荐......fail_time={fail_time}")
        fail_time += 1
        reflect_prompt = (
            "The list of actions, predicates, and objects you provided is insufficient to accomplish the specified goals: {goals}. "
            "Specifically, these only allow for the completion of the {have_finished}, while failing to address the {not_finished}.\n"

            "1. In regards to the {not_finished}, several critical dependencies and tools have been overlooked."
            " This includes the need to plug in an electrical appliance before using it, and the need to open a container before placing items inside.\n "

            '2. Additionally, for the {not_finished}, it appears that essential tools like \'rag\', \'faucet\', \'kitchenknife\' and steps have also been neglected.'
            "The requirements include using a rag for wiping, going to the faucet and turning it on for washing, wiping and washing to clean an object, and using a kitchen knife for cutting..\n"

            '3. Moreover, for the {not_finished}, some actions require coordination between the hands that was not managed correctly. For example, handling different objects simultaneously with both hands when necessary.\n"'

            "Please re-analyze the specified goal to identify the optimal actions, essential action predicates, and key objects necessary for achieving the goals. Use the same format as previously used, beginning with 'Optimal Actions:', 'Vital Action Predicates:', and 'Vital Objects:' respectively.")

            # 还有左右手问题 需要写一下

        # 获取所有叶子状态的并集，看看是哪个任务完成不了
        # algo.algo.bt
        # result_conditions = collect_conditions(algo.algo.bt)
        # have_finished = goal & result_conditions
        # not_finished = goal - result_conditions
        not_finished=set()
        # 单独跑每一个看看是哪个没完成
        for g in d["Goals"]:
            algo_tmp = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set, \
                                  priority_act_ls=priority_act_ls, key_predicates=key_predicates,
                                  key_objects=key_objects, \
                                  selected_algorithm="opt", mode="small-predicate-objs", \
                                  llm_reflect=False, time_limit=3,
                                  heuristic_choice=0)
            gset = goal_transfer_str(g)
            algo_tmp.process(gset)
            state_g = cur_cond_set

            gg = gset[0]
            error_g, state_g, _, _, _ = algo_tmp.execute_bt(gg, state_g, verbose=False)
            if error_g:
                not_finished |= gg

        have_finished = goal-not_finished

        have_finished_str = convert_conditions(have_finished)
        not_finished_str = convert_conditions(not_finished)
        # Inserting actual sets into the prompt
        reflect_prompt = reflect_prompt.format(goals=d['Goals'],have_finished=have_finished_str, not_finished=str(not_finished))

        # 基本反馈prompt模板
        # reflect_prompt = (
        #     "The list of actions, predicates, and objects you provided is insufficient to accomplish the specified goals: {}. "
        #     "Specifically, these only allow for the completion of the {}, while failing to address the {}.\n"
        # ).format(goals, have_finished_str, not_finished_str)
        #
        # # 各种未完成目标的具体提示
        # ref_prompt_dic = {
        #     'IsIn': 'For items like {object}, it appears that steps such as opening the container {container} before placing items inside have been neglected. If {object} or {container} requires power, ensure it is plugged in before use.',
        #     'SwitchOn': 'It seems that preparatory actions such as plugging in the appliance {object} have been missed. This includes ensuring that electrical appliances like {object} are plugged in before attempting to switch them on.',
        #     'IsClean': 'Essential tools and actions like using a rag for wiping or going to the faucet and turning it on for washing appear to have been overlooked. The requirements include wiping and washing {object} to ensure it is clean.',
        #     'IsCut': 'It appears that essential tools like a kitchen knife have been neglected. The requirements include using a kitchen knife to properly cut {object}.'
        # }
        #
        # # 解析未完成的目标，生成对应的提示信息
        # additional_prompts = []
        # for goal in not_finished_str:
        #     action_type = goal.split('_')[0]  # 提取动作类型
        #     details = goal.split('_')[1:]  # 提取目标细节（如对象和容器）
        #
        #     prompt_template = ref_prompt_dic.get(action_type,
        #                                          "Additional actions, predicates, and objects are needed to address the goal: {}.".format(goal))
        #
        #     # 创建一个字典用于格式化字符串
        #     format_dict = {}
        #     if len(details) > 0:
        #         format_dict['object'] = details[0]
        #     if len(details) > 1:
        #         format_dict['container'] = details[1]
        #
        #     # 检查是否有足够的信息来格式化字符串
        #     try:
        #         formatted_prompt = prompt_template.format(**format_dict)
        #         additional_prompts.append(formatted_prompt)
        #     except KeyError as e:
        #         print(f"Missing key for formatting: {e}")
        #         continue  # 如果缺少必要的键，则跳过此目标的提示
        #
        # # 将所有具体提示添加到基本反馈prompt中
        # reflect_prompt += "\n".join(additional_prompts)
        # reflect_prompt += (
        #     "\n\nPlease generate the missing actions to address these unmet goals, re-analyze the specified goal to identify the optimal actions, essential action predicates, and key objects necessary for achieving the goals. "
        #     "Use the same format as previously used, beginning with 'Optimal Actions:', 'Vital Action Predicates:', and 'Vital Objects:' respectively.")


        messages.append({"role": "user", "content": reflect_prompt})
        print("reflect_prompt:", reflect_prompt)

        # 大模型推荐的结果
        priority_act_ls_new, llm_key_pred_new, llm_key_obj_new, messages = \
            extract_llm_from_reflect(llm, messages)

        priority_act_ls =  list(OrderedSet(priority_act_ls + priority_act_ls_new))
        llm_key_pred = list(OrderedSet(llm_key_pred + llm_key_pred_new))
        llm_key_obj = list(OrderedSet(llm_key_obj + llm_key_obj_new))

        #########################
        print("Rec Act:", priority_act_ls)
        print("Rec Pred", llm_key_pred)
        print("Rec Obj:", llm_key_obj)

        # key_predicates 和 key_objects 要将推荐的 priority_act_ls 补充进来
        _, pred, obj = act_format_records(priority_act_ls)
        key_predicates = list(set(llm_key_pred + pred))
        key_objects = list(set(llm_key_obj + obj))

        algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set, \
                              priority_act_ls=priority_act_ls, key_predicates=key_predicates,
                              key_objects=key_objects, \
                              selected_algorithm="opt", mode="small-predicate-objs", \
                              llm_reflect=False, time_limit=10,
                              heuristic_choice=0)

        start_time = time.time()
        goal_set = goal_transfer_str(' & '.join(d["Goals"]))
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
        ########################





    # Add the results to the entry with appropriate prefixes
    result_entry[f'reflect'] = fail_time
    result_entry[f'Timeout'] = 1 if time_limit_exceeded == True else 0
    result_entry[f'err'] = 1 if error==True else 0
    result_entry[f'exp'] = expanded_num
    result_entry[f'time'] = planning_time_total
    result_entry[f'cost'] = cost
    result_entry[f'act'] = record_act_ls

    results.append(result_entry)


df = pd.DataFrame(results)
time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()).replace("-", "").replace(":", "")
# df.to_csv(f"LLM_40_time={time_str}.csv", index=False)
# print(f"Results have been saved to LLM_40_time={time_str}.csv")
df.to_csv(f"LLM_40_reflect.csv", index=False)
print(f"Results have been saved to LLM_40_reflect.csv")


end_time = time.time()
time_total = (end_time - start_time)
print("Total time:", time_total)

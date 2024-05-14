import pandas as pd
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

from tools import execute_algorithm,load_dataset,setup_default_env




# 主函数
llm = LLMGPT3()
default_prompt_file = f"{ROOT_PATH}\\algos\\llm_client\\prompt_VHT_just_goal.txt"
dataset = load_dataset(f"{ROOT_PATH}/../test/dataset/data1_env1_40_test_reflect.txt")

start_time = time.time()

for id, d in enumerate(dataset[:10]):
    print("\x1b[32m\n== ID:", id, "  ", d['Goals'], "\x1b[0m")
    goals = d['Goals']
    d['Optimal Actions'] = act_str_process(d['Optimal Actions'], already_split=True)


    env, cur_cond_set = setup_default_env()

    # Initial recommendation from LLM
    priority_act_ls, llm_key_pred, llm_key_obj, messages = \
                        extract_llm_from_instr_goal(llm, default_prompt_file, goals, verbose=False,
                                                    choose_database=True)
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
    expanded_num, planning_time_total, cost, error, act_num, current_cost, record_act_ls \
                                                                    = execute_algorithm(algo, goal_set, cur_cond_set)
    time_limit_exceeded = algo.algo.time_limit_exceeded

    print(f"\x1b[32mExecuted {act_num} action steps\x1b[0m",
          "\x1b[31mERROR\x1b[0m" if error else "",
          "\x1b[31mTIMEOUT\x1b[0m" if time_limit_exceeded else "")
    print("current_cost:", current_cost, "expanded_num:", expanded_num, "planning_time_total:", planning_time_total)


    # 如果成功就放入数据库
    # 将整条数据插入数据库


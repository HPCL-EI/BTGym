
from btgym.utils.tools import collect_action_nodes
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from btgym.algos.llm_client.tools import goal_transfer_str, act_format_records
from tools import *


from btgym.envs.robowaiter.exec_lib._base.VHTAction import VHTAction
env = btgym.make("RWEnv")
cur_cond_set = env.agents[0].condition_set = {'RobotNear(Bar)','Holding(Nothing)' }
cur_cond_set |= {f'Exists({arg})' for arg in VHTAction.all_object-{'Coffee', 'Water', 'Dessert'}}
cur_cond_set |= {f'Exists({arg})' for arg in VHTAction.all_object-{'Coffee', 'Water', 'Dessert'}}

print(f"共收集到 {len(VHTAction.all_object)} 个物体")



goal=['On_Water_Table1']
# goal = ['Low_ACTemperature']
goal_set = goal_transfer_str(' & '.join(goal))

algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                      priority_act_ls=[], key_predicates=[],
                      key_objects=[],
                      selected_algorithm="opt", mode="big",
                      llm_reflect=False, time_limit=30,
                      heuristic_choice=0)
expanded_num, planning_time_total, cost, error, act_num, current_cost, record_act_ls = \
    execute_algorithm(algo, goal_set, cur_cond_set)
time_limit_exceeded = algo.algo.time_limit_exceeded

success = not error and not time_limit_exceeded

_priority_act_ls, key_predicates, key_objects = act_format_records(record_act_ls)
priority_act_ls = record_act_ls
# 打印所有变量
# 定义蓝色的ANSI转义序列
BLUE = "\033[94m"
RESET = "\033[0m"

print(f"{BLUE}Try to use big space...{RESET}")

# 打印指定的三个输出，并使用蓝色
print(f"{BLUE}success:{RESET}", success)
print(f"{BLUE}goal:{RESET}", ' & '.join(goal))
print(f"{BLUE}_priority_act_ls:{RESET}", _priority_act_ls)
print(f"{BLUE}act_num:{RESET}", act_num)
print(f"{BLUE}planning_time_total:{RESET}", planning_time_total)
print(f"{BLUE}expanded_num:{RESET}", expanded_num)
print(f"{BLUE}current_cost:{RESET}", current_cost)
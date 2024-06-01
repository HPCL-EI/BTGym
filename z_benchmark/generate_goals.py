import time
import os
import re
import btgym
from btgym.utils import ROOT_PATH
from btgym.algos.llm_client.tools import goal_transfer_str, act_format_records
from btgym.algos.bt_autogen.main_interface import BTExpInterface

from btgym.utils.tools import collect_action_nodes
from btgym.utils.goal_generator.rw_gen import RoboWaiterGoalGen
from btgym.utils.goal_generator.vh_gen import VirtualHomeGoalGen
from btgym.utils.goal_generator.rhs_gen import RobotHowSmallGoalGen
from btgym.utils.goal_generator.rh_gen import RobotHowGoalGen

data_num = 100
max_goal_num=500
diffcult_type= "mix" #"single"  #"mix" "multi"
scene = "RH"

if scene=="RW":
    # ===================== RoboWaiter ========================
    goal_gen = RoboWaiterGoalGen()
    goal_ls = goal_gen.random_generate_goals(max_goal_num,diffcult_type=diffcult_type)
    for goal in goal_ls:
        print(goal)

    from btgym.envs.RoboWaiter.exec_lib._base.RWAction import RWAction

    env = btgym.make("RWEnv")
    cur_cond_set = env.agents[0].condition_set = {'RobotNear(Bar)', 'Holding(Nothing)'}
    cur_cond_set |= {f'Exists({arg})' for arg in RWAction.all_object - {'Coffee', 'Water', 'Dessert'}}
    cur_cond_set |= {f'Exists({arg})' for arg in RWAction.all_object - {'Coffee', 'Water', 'Dessert'}}
    big_actions = collect_action_nodes(env.behavior_lib)

elif scene=="VH":
    # ===================== VirtualHome ========================
    goal_gen = VirtualHomeGoalGen()
    goal_ls = goal_gen.random_generate_goals(max_goal_num,diffcult_type=diffcult_type)
    for goal in goal_ls:
        print(goal)

    from btgym.envs.virtualhome.exec_lib._base.VHAction import VHAction
    env = btgym.make("VH-PutMilkInFridge")
    cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
    cur_cond_set |= {f'IsClose({arg})' for arg in VHAction.CanOpenPlaces}
    cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHAction.HasSwitchObjects}
    big_actions = collect_action_nodes(env.behavior_lib)

elif scene=="RHS":
    # ===================== RobotHow-Small ========================
    goal_gen = RobotHowSmallGoalGen()
    goal_ls = goal_gen.random_generate_goals(max_goal_num,diffcult_type=diffcult_type)
    for goal in goal_ls:
        print(goal)

    from btgym.envs.RobotHow_Small.exec_lib._base.RHSAction import RHSAction
    env = btgym.make("VHT-Small")
    cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
    cur_cond_set |= {f'IsClose({arg})' for arg in RHSAction.CAN_OPEN}
    cur_cond_set |= {f'IsUnplugged({arg})' for arg in RHSAction.HAS_PLUG}
    cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in RHSAction.HAS_SWITCH}
    big_actions = collect_action_nodes(env.behavior_lib)

elif scene=="RH":
    # ===================== RobotHow ========================
    goal_gen = RobotHowGoalGen()
    goal_ls = goal_gen.random_generate_goals(max_goal_num,diffcult_type=diffcult_type)
    for goal in goal_ls:
        print(goal)

    from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction as RHB
    env = btgym.make("VHT-PutMilkInFridge")
    cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
    cur_cond_set |= {f'IsClose({arg})' for arg in RHB.CAN_OPEN}
    cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in RHB.HAS_SWITCH}
    cur_cond_set |= {f'IsUnplugged({arg})' for arg in RHB.HAS_PLUG}
    print(f"Collected a total of {len(RHB.AllObject)} objects")
    big_actions = collect_action_nodes(env.behavior_lib)


file_name = f"{scene}_{diffcult_type}_{data_num}"
output_path = f"{ROOT_PATH}/../z_benchmark/data/{file_name}_processed_data.txt"


def write_to_file(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a') as file:
        file.write(data + '\n')


need_cost = False

# 把env，goal，act，act pred，obj
id=0
for i,goal_str in enumerate(goal_ls):
# for i,goal_str in enumerate(['IsIn_milk_fridge']):

    print("i:",i,"goal_str:",goal_str)

    # 把关键物体装进去
    objects = {'rag','faucet','kitchenknife'}
    pattern = re.compile(r'\((.*?)\)')
    goal_set = goal_transfer_str(goal_str)
    for expr in goal_set[0]:
        match = pattern.search(expr)
        if match:
            objects.update(match.group(1).split(','))
    priority_obj_ls = list(objects)

    algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                          priority_act_ls=[], key_predicates=[],
                          key_objects=priority_obj_ls,
                          selected_algorithm="opt", mode="small-objs",
                          llm_reflect=False, time_limit=10,
                          heuristic_choice=-1)

    goal_set = goal_transfer_str(goal_str)

    start_time = time.time()
    algo.process(goal_set)
    end_time = time.time()
    planning_time_total = end_time - start_time

    time_limit_exceeded = algo.algo.time_limit_exceeded

    ptml_string, cost, expanded_num = algo.post_process()
    error, state, act_num, current_cost, record_act_ls = algo.execute_bt(goal_set[0], cur_cond_set, verbose=False)

    print(f"\x1b[32m Goal:{goal_str} \n Executed {act_num} action steps\x1b[0m",
          "\x1b[31mERROR\x1b[0m" if error else "",
          "\x1b[31mTIMEOUT\x1b[0m" if time_limit_exceeded else "")
    print("current_cost:", current_cost, "expanded_num:", expanded_num, "planning_time_total:", planning_time_total)

    if error or time_limit_exceeded:
        continue #break

    correct_act, predicate, objects = act_format_records(record_act_ls)
    def extract_and_format(items):
        return ', '.join(items)


    formatted_act = extract_and_format(correct_act)
    formatted_predicates = extract_and_format(predicate)
    formatted_objects = extract_and_format(objects)

    id += 1
    entry_str = f"{id}\n"
    entry_str += f"Environment: \n"
    entry_str += f"Instruction: \n"
    entry_str += f"Goals: {goal_str}\n"
    entry_str += f"Optimal Actions: {formatted_act}\n"
    entry_str += f"Vital Action Predicates: {formatted_predicates}\n"
    entry_str += f"Vital Objects: {formatted_objects}\n"
    if need_cost:
        entry_str += f"cost: {str(current_cost)}\n"

    write_to_file(entry_str, output_path)
    print("Written to file:", entry_str)

    if id >=data_num:
        print(f"Reached {data_num}.")
        break

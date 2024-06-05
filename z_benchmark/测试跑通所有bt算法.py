import time

import btgym
from btgym.algos.llm_client.tools import goal_transfer_str
from btgym.algos.bt_autogen.main_interface import BTExpInterface

from btgym.utils.tools import collect_action_nodes
from btgym.utils.goal_generator.vh_gen import VirtualHomeGoalGen

data_num = 100
max_goal_num=500
diffcult_type= "single" #"single"  #"mix" "multi"
scene = "VH"

# ===================== VirtualHome ========================
goal_gen = VirtualHomeGoalGen()
goal_ls = goal_gen.random_generate_goals(max_goal_num ,diffcult_type=diffcult_type)
for goal in goal_ls:
    print(goal)

from btgym.envs.virtualhome.exec_lib._base.VHAction import VHAction
env = btgym.make("VH-PutMilkInFridge")
cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
cur_cond_set |= {f'IsClose({arg})' for arg in VHAction.CanOpenPlaces}
cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHAction.HasSwitchObjects}
big_actions = collect_action_nodes(env.behavior_lib)

# for i,goal_str in enumerate(goal_ls):
# for i,goal_str in enumerate(['IsIn_milk_desk']): #  & IsOn_milk_desk
for i,goal_str in enumerate(['IsIn_milk_fridge']):
    print("i:", i, "goal_str:", goal_str)

    # selected_algorithm,mode="big"
    # 暂时不知道为什么 opt 和 obtea 的时间为什么不一样，obtea是从opt中删除启发式得来的
    # opt:        5act, current_cost: 48 expanded_num: 189 planning_time_total: 0.29288530349731445
    # obtea=opt:        current_cost: 48 expanded_num: 189 planning_time_total: 0.3274109363555908
    # bfs=bt-Exp: 8act, current_cost: 83 expanded_num: 157 planning_time_total: 0.6710596084594727
    # dfs: xxx
    # weakAlgo: 经常失败？

    # mode="small-predicate-objs"
    # opt: 5act, current_cost: 48 expanded_num: 189 planning_time_total: 0.29288530349731445
    # h0:  5act, current_cost: 48 expanded_num: 9 planning_time_total: 0.000997304916381836
    # h1:  5act, current_cost: 48 expanded_num: 9 planning_time_total: 0.001996278762817383
    # act_str = ['Walk_milk', 'RightGrab_milk', 'Walk_fridge', 'Open_fridge',\
    #     'RightPutIn_milk_fridge']
    # priority_act_ls = act_str_process(act_str,already_split=True)
    # key_predicates = ['Walk', 'RightGrab', 'Open', 'RightPutIn']
    # key_objects=['milk', 'fridge']

    algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                          priority_act_ls=[], key_predicates=[],
                          key_objects=[],
                          selected_algorithm="opt", mode="big",
                          llm_reflect=False, time_limit=15,
                          heuristic_choice=0,output_just_best=True)

    goal_set = goal_transfer_str(goal_str)

    start_time = time.time()
    algo.process(goal_set)
    end_time = time.time()
    planning_time_total = end_time - start_time

    time_limit_exceeded = algo.algo.time_limit_exceeded

    ptml_string, cost, expanded_num = algo.post_process()
    error, state, act_num, current_cost, record_act_ls,ticks = algo.execute_bt(goal_set[0], cur_cond_set, verbose=False)

    print(f"\x1b[32m Goal:{goal_str} \n Executed {act_num} action steps\x1b[0m",
          "\x1b[31mERROR\x1b[0m" if error else "",
          "\x1b[31mTIMEOUT\x1b[0m" if time_limit_exceeded else "")
    print("current_cost:", current_cost, "expanded_num:", expanded_num, "planning_time_total:", planning_time_total)

# visualization
file_name = "tree"
file_path = f'./{file_name}.btml'
with open(file_path, 'w') as file:
    file.write(ptml_string)
# read and execute
from btgym import BehaviorTree
bt = BehaviorTree(file_name + ".btml", env.behavior_lib)
# bt.print()
bt.draw()
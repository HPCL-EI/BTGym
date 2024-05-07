import pandas as pd
import btgym
import time
from btgym import BehaviorTree
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction

# Load the results from the previously saved CSV file
# file_name = "llm_40.csv"
file_name = "llm_40_just_goal_gpt4.csv"
results_df = pd.read_csv(file_name)
results = results_df.to_dict(orient='records')  # Convert DataFrame back to list of dictionaries

# Check that the data is successfully loaded
print(f"Loaded {len(results)} results from '{file_name}'")


data = results[0]
goal_set =
# goal_set = [{'IsIn(cupcake,fridge)'}]
priority_act_ls=
key_pred=
key_obj=


env = btgym.make("VHT-PutMilkInFridge")
cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
cur_cond_set |= {f'IsClose({arg})' for arg in VHTAction.CAN_OPEN}
cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHTAction.HAS_SWITCH}
cur_cond_set |= {f'IsUnplugged({arg})' for arg in VHTAction.HAS_PLUG}

algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set, \
                      priority_act_ls=priority_act_ls,key_predicates=key_pred, key_objects=key_obj, \
                      selected_algorithm="opt", mode="small-predicate-objs", \
                      llm_reflect=False)

start_time = time.time()
algo.process(goal_set)
end_time = time.time()

ptml_string, cost, expanded_num = algo.post_process()  # 后处理
print("Expanded Conditions: ", expanded_num)
planning_time_total = (end_time - start_time)
print("planning_time_total:", planning_time_total)
print("cost_total:", cost)

# simulation and test
print("\n================ ")
goal = goal_set[0]
state = cur_cond_set
error, state, act_num, current_cost, record_act_ls = algo.execute_bt(goal, state, verbose=True)
print(f"一定运行了 {act_num - 1} 个动作步")
print("current_cost:", current_cost)
print("================ ")
import time

from btgym import BehaviorTree
from btgym import ExecBehaviorLibrary
import btgym

from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from btgym.algos.bt_autogen.Action import Action

# behavior_tree = BehaviorTree("Default.btml")
# behavior_tree.print()
# behavior_tree.draw()

# lib_path = f'{btgym.ROOT_PATH}/exec_lib'
# exec_lib = ExecBehaviorLibrary(lib_path)
# print(exec_lib.Action)

# exec_bt = ExecBehaviorTree("Default.btml",exec_lib)

def collect_action_nodes():
    action_list = []
    lib_path = f'{btgym.ROOT_PATH}/envs/virtualhome/exec_lib'
    behavior_dict = ExecBehaviorLibrary(lib_path)
    for cls in behavior_dict["Action"].values():
        if cls.can_be_expanded:
            print(f"可扩展动作：{cls.__name__}, 存在{len(cls.valid_args)}个有效论域组合")
            if cls.num_args == 0:
                action_list.append(Action(name=cls.get_ins_name(), **cls.get_info()))
            if cls.num_args == 1:
                for arg in cls.valid_args:
                    action_list.append(Action(name=cls.get_ins_name(arg), **cls.get_info(arg)))
            if cls.num_args > 1:
                for args in cls.valid_args:
                    action_list.append(Action(name=cls.get_ins_name(*args), **cls.get_info(*args)))
    return action_list



env = btgym.make("VH-PutMilkInFridge")


#todo: LLMs

# question="请把厨房桌上的牛奶放到冰箱里,并关上冰箱。"
question="请打开电视"
llm = LLMGPT3()
prompt=""
# goal = llm.request(question=prompt+question)
# goal=[{"IsWatching(self,tv)"}]
# goal=[{"IsIn(milk,fridge)","IsClosed(fridge)"}]  # goal 是 set组成的列表

# goal=[{"IsIn(milk,fridge)"}]
goal=[{"IsOpened(fridge)"}]
print("goal",goal)

#todo: BTExp
#todo: BTExp:LoadActions
# action_list=None
action_list = collect_action_nodes()
print(f"共收集到{len(action_list)}个实例化动作:")
# for a in self.action_list:
#     if "Turn" in a.name:
#         print(a.name)
print("--------------------\n")

#todo: BTExp:process
cur_cond_set=env.agents[0].condition_set = {"IsSwitchedOff(tv)","IsClosed(fridge)",
                               "IsRightHandEmpty(self)","IsLeftHandEmpty(self)","IsStanding(self)"
                               }

algo = BTExpInterface(action_list, cur_cond_set)
ptml_string = algo.process(goal)

file_name = "grasp_milk"
file_path = f'./{file_name}.btml'
with open(file_path, 'w') as file:
    file.write(ptml_string)



# 读取执行
bt = BehaviorTree("grasp_milk.btml", env.behavior_lib)
bt.print()

env.agents[0].bind_bt(bt)
env.reset()

is_finished = False
while not is_finished:
    is_finished = env.step()
    # print(env.agents[0].condition_set)

    g_finished=True
    for g in goal:
        if not g<= env.agents[0].condition_set:
            g_finished=False
    if g_finished:
        is_finished=True
env.close()
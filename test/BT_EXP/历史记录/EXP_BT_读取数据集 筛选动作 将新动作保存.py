import btgym
from btgym.utils import ROOT_PATH
# from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
# from btgym.envs.virtualhometext.exec_lib._base.VHTAction_small import VHTAction_small
from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction

import random
import numpy as np

seed = 0
random.seed(seed)
np.random.seed(seed)

from btgym.utils.read_dataset import read_dataset
from btgym.utils.tools import collect_action_nodes

import pickle

# def refresh_VHT_samll_data():
# 读入数据集合
data_path = f"{ROOT_PATH}/../test/dataset/data0429.txt"
data = read_dataset(data_path)
data_num = len(data)
print(f"导入 {data_num} 条数据")
print(data[0])

# 数据集中涉及的所有物体集合
objs=set()
for d in data:
    objs |= set(d['Key_Object'])

categories = ['SURFACES', 'SITTABLE', 'CAN_OPEN', 'CONTAINERS', 'GRABBABLE', 'cleaning_tools', \
         'cutting_tools', 'HAS_SWITCH', 'HAS_PLUG', 'CUTABLE', 'EATABLE', 'WASHABLE', 'RECIPIENT', \
         'POURABLE', 'DRINKABLE']
categories_objs_dic={}
for ctg in categories:
    categories_objs_dic[ctg] = getattr(VHTAction, ctg)
    categories_objs_dic[ctg] &= objs


ctg_objs_path = f"{ROOT_PATH}/../test/EXP/ctg_objs.pickle"
# 打开一个文件用于写入，注意'b'表示二进制模式
with open(ctg_objs_path, 'wb') as file:
    # 使用pickle.dump()函数将数据写入文件
    pickle.dump(categories_objs_dic, file)


env = btgym.make("VHT-Small")
print(env.behavior_lib['Action'])
action_list = collect_action_nodes(env.behavior_lib)




# behavior_lib_path = f"{ROOT_PATH}/../test/EXP/bt_exec_lib.pickle"
# env.reload_behavior_lib(behavior_lib_path)
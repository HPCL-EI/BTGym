import copy
import time

from btgym import BehaviorTree
import btgym
from btgym.utils import ROOT_PATH
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from btgym.envs.virtualhometext.exec_lib._base.VHTAction_small import VHTAction_small
from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction
from btgym.algos.bt_autogen.tools import state_transition
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process
import random
import numpy as np
import pandas as pd

seed = 0
random.seed(seed)
np.random.seed(seed)

from btgym.utils.tools import collect_action_nodes
from btgym.utils.read_dataset import read_dataset
from tools import count_accuracy, identify_and_print_diffs, analyze_data_tabular
import pickle
all_start_time = time.time()




# 导入数据
data_path = f"./hard_test_20_processed_data.txt" #DATA_BT_100_ori_yz_revby_cys
data = read_dataset(data_path)
len_data = len(data)
print(f"导入 {len_data} 条数据")
print(data[0])
analyze_data_tabular(data,[17,1,1,1])
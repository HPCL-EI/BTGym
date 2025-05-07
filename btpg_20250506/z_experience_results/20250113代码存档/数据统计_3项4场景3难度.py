import os
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter
from btgym.utils import ROOT_PATH
from btgym.utils.read_dataset import read_dataset
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process, act_format_records
from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction
from btgym.envs.RobotHow_Small.exec_lib._base.OGAction import OGAction
from btgym.envs.RoboWaiter.exec_lib._base.RWAction import RWAction
from btgym.envs.VirtualHome.exec_lib._base.VHAction import VHAction
os.chdir(f'{ROOT_PATH}/../z_experience_results')

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['mathtext.fontset'] = 'stix'

scene_Type = {}
scene_Type["RH"] = {'SURFACES':RHAction.SURFACES,'SITTABLE':RHAction.SITTABLE,'CAN_OPEN':RHAction.CAN_OPEN,
                    'CONTAINERS':RHAction.CONTAINERS,'GRABBABLE':RHAction.GRABBABLE,'HAS_SWITCH':RHAction.HAS_SWITCH,
                    'HAS_PLUG':RHAction.HAS_PLUG,'CUTABLE':RHAction.CUTABLE,'WASHABLE':RHAction.WASHABLE,
                    }
scene_Type["RHS"] = {'SURFACES':OGAction.SURFACES,'SITTABLE':OGAction.SITTABLE,'CAN_OPEN':OGAction.CAN_OPEN,
                    'CONTAINERS':OGAction.CONTAINERS,'GRABBABLE':OGAction.GRABBABLE,'HAS_SWITCH':OGAction.HAS_SWITCH,
                    'HAS_PLUG':OGAction.HAS_PLUG,'CUTABLE':OGAction.CUTABLE,'WASHABLE':OGAction.WASHABLE,
                    }
scene_Type["RW"] = {'SURFACES':RWAction.SURFACES,'GRABBABLE':RWAction.GRABBABLE}
scene_Type["VH"] = {'SURFACES':VHAction.SURFACES,'SITTABLE':VHAction.SITTABLE,'CAN_OPEN':VHAction.CAN_OPEN,
                    'CONTAINERS':VHAction.CONTAINERS,'GRABBABLE':VHAction.GRABBABLE,'HAS_SWITCH':VHAction.HAS_SWITCH}


def plot_hist(plot_type,difficulty):
    for scene in ['RH', 'RHS', 'RW', 'VH']:
        # 导入数据
        data_path = f"{ROOT_PATH}/../z_experience_results/data/{scene}_{difficulty}_100_processed_data.txt"
        data = read_dataset(data_path)

        statistic = []  # 统计量
        for i, d in enumerate(data):
            goal_str = ' & '.join(d["Goals"])
            goal_set = goal_transfer_str(goal_str)
            if plot_type == 'Actions':
                statistic += [len(act_str_process(d['Optimal Actions'], already_split=True))]
            if plot_type == 'Predicates':
                statistic += d['Vital Action Predicates']
            if plot_type == 'Objects':  # Obj先统计物品数量
                statistic += d['Vital Objects']

        counts = Counter(statistic)  # 计数

        if plot_type == 'Objects':   # Obj再统计物品属于哪些属性
            type_counts = {key: 0 for key in scene_Type[scene].keys()}
            for item, count in counts.items():
                for attribute, items_set in scene_Type[scene].items():
                    if item in items_set:
                        type_counts[attribute] += count
            counts = type_counts

        counts_deorder = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
        plt.figure(figsize=(10, 6))

        # keys = list(counts_deorder.keys())
        # values = list(counts_deorder.values())
        # keys = np.array(list(counts_deorder.keys()))
        # values = np.array(list(counts_deorder.values()))
        # print("key:",keys)
        # print("values:",values)
        # if plot_type == 'Actions':
        #     keys = ['num='+str(key) for key in keys]
        # print("key:",keys)
        # print("values:",values)
        # print("counts_deorder.keys:", counts_deorder.keys())
        # print("counts_deorder.values:", counts_deorder.values())
        # key_ls = list(counts_deorder.keys())
        # value_ls = list(counts_deorder.values())

        print(counts_deorder.keys())
        print(counts_deorder.values())
        plt.bar(counts_deorder.keys(), counts_deorder.values(), color='skyblue')
        plt.xlabel(f'Number of {plot_type}')
        plt.ylabel('Counts')
        plt.title(f'Histogram of Lengths of {plot_type} in {scene} of {difficulty}')
        plt.xticks(list(counts.keys()))
        plt.savefig(f'./images_histogram/{plot_type}_{scene}_{difficulty}.png',dpi=100)
        plt.show()

plot_type = 'Actions'   # 'Actions', 'Predicates', 'Objects'
difficulty = 'single'   # 'single', 'mix', 'multi'

for plot_type in ['Actions', 'Predicates', 'Objects']:  # 'Actions', 'Predicates', 'Objects'
    for difficulty in ['single', 'multi']:  # 'single', 'mix', 'multi'
        plot_hist(plot_type,difficulty)


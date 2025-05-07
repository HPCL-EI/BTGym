import matplotlib
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from btgym.utils import ROOT_PATH
import pandas as pd

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 22
matplotlib.rcParams['mathtext.fontset'] = 'stix'

def plot_ratio(difficulty, scene, ax):
    file_path = f'{ROOT_PATH}/../z_experience_results/output_algo_act_num/{scene}_{difficulty}_maxep={maxep}_act_num.csv'
    data = pd.read_csv(file_path)
    
    # hbtp_file_path = f'{ROOT_PATH}/../z_experience_results/output_algo_act_num/hbtp_{scene}_{difficulty}_maxep={maxep}_act_num.csv'
    hbtp_file_path = f'{ROOT_PATH}/../z_experience_results/output_algo_act_num/{scene}_{difficulty}_maxep={maxep}_act_num.csv'
    hbtp_data = pd.read_csv(hbtp_file_path)

    sorted_x = []
    counters = []
    for algo_str in algo_type:
        if algo_str == "hbtp":
            data_algo = hbtp_data[algo_str].dropna().tolist()
        else:
            data_algo = data[algo_str].dropna().tolist()
        counter = Counter(data_algo)
        if len(counter.keys()) > len(sorted_x):  # 获得完整的x轴范围
            sorted_x = sorted(counter.keys())
        counters.append(counter)
    # 设置柱状图的宽度和位置
    bar_width = 0.16
    x_algo = []  # 存储4个算法的x和y
    y_algo = []
    for i in range(len(algo_type)):
        if i == 0:
            x_algo.append(range(len(sorted_x)))
        else:
            x_algo.append([x + bar_width for x in x_algo[-1]])
        y_algo.append([counters[i][x] if x in counters[i] else 0 for x in sorted_x])

    for i in range(len(algo_type)):
        ax.bar(x_algo[i], y_algo[i], width=bar_width, label=algo_map[algo_type[i]], alpha=alpha)
        ax.set_xticks([x + bar_width for x in range(len(sorted_x))], sorted_x)
        ax.set_xticklabels([f'{int(x)}' for x in sorted_x])
        ax.set_title(f'{scenes2name[scene]}')
        ax.set_xlabel('Region Distance')
        ax.set_ylabel('Frequency')


scenes = ['RW','VH','RHS','RH']  # 'RW','VH','RHS','RH'
scenes2name = {'RW':'RoboWaiter', 'VH':'VirtualHome', 'RHS':'OmniGibson', 'RH':'Headless'}
difficulties=['multi']  # 'single','multi'
algo_type = ['bfs','obtea','opt_h0_llm','hbtp','opt_h0'] # 'opt_h0','opt_h0_llm','obtea','bfs'
algo_map = {'bfs':'BT Expansion', 'obtea':'OBTEA', 'opt_h0_llm':'HBTP', 'opt_h0':'HBTP-Oracle', 'hbtp':'UHBTP'}
maxep = 10
alpha = 0.60

fig, axes = plt.subplots(len(difficulties), len(scenes), figsize=(20, 5))
axes = axes.flatten()
axe_id = 0
for difficulty in difficulties:
    for scene in scenes:
        plot_ratio(difficulty, scene, axes[axe_id])
        axe_id += 1
        # Load the CSV file
handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', ncol=len(algo_type), bbox_to_anchor=(0.5, -0.04), frameon=False)
# plt.subplots_adjust(bottom=0.3)

fig.legend(handles, labels, loc='lower center', ncol=len(algo_type), bbox_to_anchor=(0.5, -0.06), frameon=False)
plt.subplots_adjust(bottom=0.1)

plt.tight_layout()
plt.savefig(f'{ROOT_PATH}/../z_experience_results/output_algo_act_num/{scene}_{difficulty}_maxep{maxep}.pdf',
            dpi=200, bbox_inches='tight', format='pdf')
plt.show()
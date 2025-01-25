import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
from btgym.utils import ROOT_PATH

# Configure the matplotlib settings
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 24
matplotlib.rcParams['mathtext.fontset'] = 'stix'

# Define fonts for the plot
font1 = {'family': 'Times New Roman', 'color': 'Black', 'weight': 'bold', 'size': 14}
font2 = {'family': 'Times New Roman', 'size': 24, 'weight': 'bold'}
font3 = {'family': 'Times New Roman', 'color': 'Black', 'weight': 'bold', 'size': 12}


def plot_percentage(percentages_type, difficulty, scene, algo_type,
                    max_epoch, y_bottom, y_top, colors, ax, y_label=False):
    if scene == 'RW':
        y_bottom = 80
        max_epoch = 40
    # if scene == 'RHS':
    #     y_bottom = 60
    if scene in ['VH', 'RHS','RH']:
        # y_bottom = 80
        max_epoch = 250
    if difficulty == 'single':
        max_epoch = 50
        y_bottom = 75

    mean_corr_ratio = []  # To store the mean of each algorithm
    std_corr_ratio = []  # To store the standard deviation of each algorithm

    for algo_str in algo_type:
        file_path = f'{ROOT_PATH}/../z_benchmark/percentage_output/{percentages_type}_{difficulty}_{scene}_{algo_str}.csv'
        df = pd.read_csv(file_path)

        corr_ratio_all = np.array(df.iloc[:, :max_epoch])   # All epochs' ratio for each data point
        mean_corr_ratio.append(list(np.mean(corr_ratio_all, axis=0)))  # Mean of each epoch
        std_corr_ratio.append(list(np.std(corr_ratio_all, axis=0)))  # Standard deviation of each epoch

    mean_corr_ratio = np.array(mean_corr_ratio)
    std_corr_ratio = np.array(std_corr_ratio)
    epochs = np.arange(1, max_epoch + 1)

    # algo_type = ['UHBTP','HOBTEA-Oracle', 'HOBTEA', 'OBTEA', 'BT Expansion']
    algo_type = ['UHBTP','HBTP-Oracle', 'HBTP', 'OBTEA', 'BT Expansion']

    index = ['RW', 'VH', 'RHS', 'RH'].index(scene)
    # scene = ['RoboWaiter', 'VirtualHome', 'RobotHow-Small', 'RobotHow'][index]
    scene = ['RoboWaiter', 'VirtualHome', 'OmniGibson', 'Headless'][index]


    for i, algo_str in enumerate(algo_type):
        # # 误差线
        # # plt.errorbar(epochs, mean_corr_ratio, yerr=std_corr_ratio, fmt='-o', capsize=5, label='Mean with Std Dev')
        # # 误差范围
        ax.plot(epochs, mean_corr_ratio[i], label=f'{algo_str}', color=colors[i]) # color=color[i],
        ax.fill_between(epochs, mean_corr_ratio[i] - var_small * std_corr_ratio[i], mean_corr_ratio[i] + var_small * std_corr_ratio[i],
                         alpha=0.08, color=colors[i])  # , label=f'{algo_str} Std Dev', color=color[i],

    if y_label:
        if difficulty == 'single':
            difficulty = 'PP in single-goal set'
        if difficulty == 'multi':
            difficulty = 'PP in multi-goal set'
        ax.set_ylabel(f'{difficulty}')
    ax.set_ylim(bottom=y_bottom,top=y_top)
    ax.set_xlabel('Exploration number')
    ax.set_title(f'{scene}')
    # ax.set_title(f'{percentages_type} ratio in {scene}')

    # ax.grid(True)

var_small = 0.2
single_max_epoch = 150
multi_max_epoch = 300
y_bottom = 50
y_top = 105
algo_type = ['hbtp','opt_h0', 'opt_h0_llm', 'obtea', 'bfs']  # Define the algorithms to plot //'opt_h0', 'opt_h0_llm', 'obtea', 'bfs'
difficulties = ['single', 'multi']
scenes = ['RW', 'VH', 'RHS', 'RH']  # 'RH', 'RHS', 'RW', 'VH'
colors = ['purple','red', 'orange', 'green', 'blue']  # orange  brown

fig, axes = plt.subplots(len(difficulties), len(scenes), figsize=(23, 11.5))
axes = axes.flatten()

axe_id = 0
for percentages_type in ['expanded']:  # 'expanded', 'traversed', 'cost'
    for difficulty in difficulties:  # 'single', 'multi'
        if difficulty == 'single': max_epoch = single_max_epoch
        elif difficulty == 'multi': max_epoch = multi_max_epoch
        for scene in scenes:  # Iterate over scenes
            if axe_id in [0,4]: y_label = True
            else: y_label = False
            plot_percentage(percentages_type, difficulty, scene,algo_type,
                            max_epoch, y_bottom, y_top, colors, axes[axe_id],
                            y_label=y_label)
            axe_id += 1

handles, labels = axes[0].get_legend_handles_labels()
# plt.subplots_adjust(left=0.1, right=0.2, bottom=0.1, top=0.2)
fig.legend(reversed(handles), reversed(labels), loc='lower center', ncol=len(algo_type), bbox_to_anchor=(0.5, -0.0175), frameon=False)
plt.subplots_adjust(bottom=0.3)
plt.tight_layout(pad=1.8, w_pad=1.0, h_pad=1.0)
# plt.tight_layout()
plt.savefig(f'{ROOT_PATH}/../z_benchmark/percentage_images/{percentages_type}_20250110.png',
            dpi=100, bbox_inches='tight')
plt.show()



#
# fig, axs = plt.subplots(2, 4, figsize=(15, 8))
#
# # 生成并绘制一些示例数据
# for i in range(2):
#     for j in range(4):
#         # 示例数据：正弦波
#         x = np.linspace(0, 10, 100)
#         y = np.sin(x + (i * 4 + j))
#         axs[i, j].plot(x, y)
#         axs[i, j].set_title(f'Subplot {i+1},{j+1}')
#
# # 调整布局，确保子图不会重叠
# plt.tight_layout()
#
# # 显示图片
# plt.show()

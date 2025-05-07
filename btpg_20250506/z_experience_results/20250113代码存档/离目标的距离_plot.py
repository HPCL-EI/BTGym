import os
import matplotlib.pyplot as plt
from btgym.utils import ROOT_PATH
import pandas as pd
import numpy as np
import matplotlib

os.chdir(f'{ROOT_PATH}/../z_experience_results')

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 30
matplotlib.rcParams['mathtext.fontset'] = 'stix'

font1 = {'family': 'Times New Roman','color': 'Black','weight': 'bold','size': 40} #normal
font2 = {'family': 'Times New Roman','size': 26,'weight': 'bold'}
font3 = {'family': 'Times New Roman','color': 'Black','weight': 'bold','size': 38}

def plot_percentage(percentages_type, difficulty, scene, algo_type, max_epoch):
    mean_corr_ratio = []  # 存储5个算法下的mean
    std_corr_ratio = []  # 存储5个算法下的std
    for algo_str in algo_type:
        file_path = f'./percentage_output/{percentages_type}_{difficulty}_{scene}_{algo_str}.csv'
        df = pd.read_csv(file_path)

        corr_ratio_all = np.array(df.iloc[:, :max_epoch])   # 每个data的所有epoch的ratio
        print(algo_str,len((list(np.mean(corr_ratio_all, axis=0)))))
        mean_corr_ratio.append(list(np.mean(corr_ratio_all, axis=0)))  # epoch的平均mean
        std_corr_ratio.append(list(np.std(corr_ratio_all, axis=0)))  # epoch的平均std

    # print(algo_str)
    # print(corr_ratio_all.shape)
    # print(len(mean_corr_ratio))
    # print(len(mean_corr_ratio[0]))
    # print(std_corr_ratio.shape,'\n')
    mean_corr_ratio = np.array(mean_corr_ratio)
    std_corr_ratio = np.array(std_corr_ratio)
    epochs = np.arange(1, max_epoch + 1)

    plt.figure(figsize=(12, 8))
    for i, algo_str in enumerate(algo_type):
        # # 误差线
        # # plt.errorbar(epochs, mean_corr_ratio, yerr=std_corr_ratio, fmt='-o', capsize=5, label='Mean with Std Dev')
        # # 误差范围
        plt.plot(epochs, mean_corr_ratio[i], label=f'{algo_str}') # color=color[i],
        plt.fill_between(epochs, mean_corr_ratio[i] - var_small * std_corr_ratio[i], mean_corr_ratio[i] + var_small * std_corr_ratio[i],
                         alpha=0.2)  # , label=f'{algo_str} Std Dev', color=color[i],

    plt.xlabel('Epoch')
    plt.ylim(bottom=y_bottom, top=y_top)
    plt.ylabel(f'{percentages_type} ratio')
    plt.title(f'{percentages_type} ratio in {scene} ({difficulty})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./percentage_images/{percentages_type}_{difficulty}_{scene}.png',
                dpi=100, bbox_inches='tight')
    plt.show()

var_small = 0.4
max_epoch = 10
y_bottom = 6
y_top = 15
algo_type = ['opt_h0','opt_h0_llm', 'obtea', 'bfs']   # 'opt_h0','opt_h0_llm', 'obtea', 'bfs',      'opt_h1','weak'

for percentages_type in ['cost_act_num_ratio']:  # 'expanded', 'traversed', 'cost'
    for difficulty in ['multi']:  # 'single', 'multi'
        print(f"============ percentages_type = {percentages_type}, difficulty = {difficulty} =============")
        for scene in ['RHS', 'RW', 'VH']:  # 'RH', 'RHS', 'RW', 'VH'
            print(f"++++++++++ scene = {scene} ++++++++++")
            plot_percentage(percentages_type, difficulty, scene, algo_type, max_epoch)





import os
import matplotlib.pyplot as plt
from btgym.utils import ROOT_PATH
import pandas as pd
import numpy as np
os.chdir(f'{ROOT_PATH}/../z_benchmark')


def plot_percentage(percentages_type, difficulty, scene, algo_type, max_epoch):
    mean_corr_ratio = []  # 存储5个算法下的mean
    std_corr_ratio = []  # 存储5个算法下的std
    for algo_str in algo_type:
        file_path = f'./output_percentage/{percentages_type}_{difficulty}_{scene}_{algo_str}.csv'
        df = pd.read_csv(file_path)

        corr_ratio_all = np.array(df)   # 每个data的所有epoch的ratio
        mean_corr_ratio.append(list(np.mean(corr_ratio_all, axis=0)))  # epoch的平均mean
        std_corr_ratio.append(list(np.std(corr_ratio_all, axis=0)))  # epoch的平均std

    mean_corr_ratio = np.array(mean_corr_ratio)
    std_corr_ratio = np.array(std_corr_ratio)
    epochs = np.arange(1, max_epoch + 1)

    plt.figure(figsize=(10, 6))
    for i, algo_str in enumerate(algo_type):
        # # 误差线
        # # plt.errorbar(epochs, mean_corr_ratio, yerr=std_corr_ratio, fmt='-o', capsize=5, label='Mean with Std Dev')
        # # 误差范围
        plt.plot(epochs, mean_corr_ratio[i], label=f'{algo_str}') # color=color[i],
        plt.fill_between(epochs, mean_corr_ratio[i] - std_corr_ratio[i], mean_corr_ratio[i] + std_corr_ratio[i],
                         alpha=0.2)  # , label=f'{algo_str} Std Dev', color=color[i],

    plt.xlabel('Epoch')
    plt.ylabel(f'{percentages_type} ratio')
    plt.title(f'{percentages_type} ratio in {scene} ({difficulty})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./images_percentage/{percentages_type}_{difficulty}_{scene}.png', dpi=100)
    plt.show()

max_epoch = 1000
algo_type = ['opt_h0','obtea']   # 'opt_h0','opt_h0_llm', 'obtea', 'bfs',      'opt_h1','weak'

for percentages_type in ['expanded']:  # 'expanded', 'traversed', 'cost'
    for difficulty in ['single']:  # 'single', 'multi'
        print(f"============ percentages_type = {percentages_type}, difficulty = {difficulty} =============")
        for scene in ['VH']:  # 'RH', 'RHS', 'RW', 'VH'
            print(f"++++++++++ scene = {scene} ++++++++++")
            plot_percentage(percentages_type, difficulty, scene, algo_type, max_epoch)





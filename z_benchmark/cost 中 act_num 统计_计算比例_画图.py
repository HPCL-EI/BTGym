from btgym.utils import ROOT_PATH
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
os.chdir(f'{ROOT_PATH}/../z_benchmark')
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['mathtext.fontset'] = 'stix'

def plot_adaptive_histograms_and_save(difficulty, scene):
    algo_names = ['opt_h0', 'opt_h0_llm', 'bfs']
    data_frames = []

    # 读取所有算法的CSV文件并存储在data_frames列表中
    for algo_name in algo_names:
        filename = f"./COST_output/{difficulty}_{scene}_{algo_name}.csv"
        df = pd.read_csv(filename)
        df['algo_name'] = algo_name
        df['cost_ratio'] = df['algo_cost'] / df['obtea_cost']
        data_frames.append(df)

    # 合并所有数据
    combined_df = pd.concat(data_frames)

    # 计算自适应bins
    min_cost = combined_df['obtea_cost'].min()
    max_cost = combined_df['obtea_cost'].max()
    bins = np.arange(min_cost, max_cost + 10, 10)

    # 定义颜色和位置偏移
    # colors = {
    #     'opt_h0': '#1f77b4',  # 蓝色
    #     'opt_h0_llm': '#2ca02c',  # 绿色
    #     'bfs': '#ff7f0e'  # 橙色
    # }
    colors = {
        'opt_h0': '#C5E0B4',  # 柔和的蓝绿色
        'opt_h0_llm': '#DAE3F3',  # 柔和的蓝色
        'bfs': '#FFD966'  # 柔和的橙色
    }
    offsets = {
        'opt_h0': -0.2,
        'opt_h0_llm': 0.0,
        'bfs': 0.2
    }

    plt.figure(figsize=(12, 8))

    # 存储每个类别下三种算法的平均数据
    average_data = []

    for algo_name in algo_names:
        algo_df = combined_df[combined_df['algo_name'] == algo_name].copy()
        algo_df.loc[:, 'group'] = pd.cut(algo_df['obtea_cost'], bins=bins, right=False)

        # 按组计算平均成本比例
        grouped = algo_df.groupby('group', observed=True)['cost_ratio'].mean().dropna()

        if not grouped.empty:
            x = np.arange(len(grouped.index)) + offsets[algo_name]
            plt.bar(x, grouped, width=0.4, color=colors[algo_name], label=algo_name, align='center')
            # 添加平均数据到列表
            for group, value in grouped.items():
                average_data.append((group, algo_name, value))

    plt.title(f'Combined Cost Ratio Histograms for {scene} in {difficulty}')
    plt.xlabel('Cost Group')
    plt.ylabel('Average Cost Ratio')
    y_min = 0
    y_max = 2
    plt.ylim(y_min, y_max)  # 设置y轴范围
    plt.xticks(np.arange(len(grouped.index.categories)), [str(g) for g in grouped.index.categories], rotation=45)
    plt.legend()

    # 创建输出目录（如果不存在）
    output_dir = "./COST_Histograms"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存图片到文件
    output_filename = f"{output_dir}/{difficulty}_{scene}_combined.png"
    plt.savefig(output_filename)
    plt.show()  # 显示图表
    plt.close()  # 关闭图表以释放内存
    print(f"直方图已保存为：{output_filename}")

    # # 输出每个类别下三种算法的平均数据
    # print("\n每个类别下三种算法的平均数据：")
    # for group, algo_name, value in average_data:
    #     print(f"类别: {group}, 算法: {algo_name}, 平均成本比例: {value:.2f}")


for difficulty in ['single']:  # 'single', 'multi'
    for scene in ['RH']:  # 'RH', 'RHS', 'RW', 'VH'
        plot_adaptive_histograms_and_save(difficulty, scene)


import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

# 文件名和路径
# base_file_name = 'new_EXP_2_output_summary_bt_data_small_100_bigerror_heuristic=1_seed='
# file_path = "./new_EXP_2_output/"
# seeds = range(10)

base_file_name = 'EXP_2_output_summary_bt_data_small_100_bigerror_heuristic=0_seed='
file_path = "exp_output_100/"
seeds = range(0,50)

# 初始化空的DataFrame用于合并数据
df_list = []

# 读取每个文件并合并数据
for seed in seeds:
    file_name = file_path + base_file_name + str(seed) + '.csv'
    df_seed = pd.read_csv(file_name)
    df_list.append(df_seed)

# 合并所有数据
df = pd.concat(df_list)

# 列表中的三个y_name分别绘制图表
# y_names = ["Total Current Cost", "Total Expanded Num", "Total Planning Time Total"]
y_names = ["Total Current Cost"]

# 是否平滑
smooth = False

for y_name in y_names:
    if not smooth:
        # Plotting
        # 无平滑处理
        fig, ax = plt.subplots(figsize=(10, 6))
        # Unique error rates to plot
        error_rates = df['Error Rate'].unique()
        for err_rate in error_rates:
            subset = df[df['Error Rate'] == err_rate]
            if err_rate==0.5:
                continue
            # 计算平均值
            mean_data = subset.groupby('Correct Rate')[y_name].mean().reset_index()
            print("err_rate:",err_rate,"mean_data:",mean_data)
            ax.plot(mean_data['Correct Rate'], mean_data[y_name], marker='o', label=f'Error Rate = {err_rate*100}%')
        ax.set_xlabel('Correct Rate')
        ax.set_ylabel(y_name)
        ax.set_title(f'{y_name} vs Correct Rate for Different Error Rates')
        ax.legend()
        plt.grid(True)
        plt.savefig(f'plot1_total_{base_file_name}_{y_name}.png')  # Save the first plot
        plt.show()
    else:
        # 绘图设置
        fig, ax = plt.subplots(figsize=(10, 6))

        # 不同的错误率
        error_rates = df['Error Rate'].unique()
        for err_rate in error_rates:
            if err_rate==0.5:
                continue
            subset = df[df['Error Rate'] == err_rate]
            if len(subset) > 3:  # 确保每个子集有足够的点
                # 数据点
                x = subset['Correct Rate']
                y = subset[y_name]
                # 使用滑动窗口平滑数据
                window_size = 3  # 滑动窗口大小设为3
                y_smoothed = y.rolling(window=window_size, center=True).mean()  # 计算移动平均，center=True表示窗口中心对齐当前值
                # 除去NaN值
                valid_indices = ~y_smoothed.isna()
                x = x[valid_indices]
                y_smoothed = y_smoothed[valid_indices]

                # 移除重复的x值并平均相应的y值
                x_unique, index = np.unique(x, return_index=True)
                y_unique = [y_smoothed[x == xi].mean() for xi in x_unique]

                # 创建样条插值对象
                if len(x_unique) > 3:  # 确保有足够的点进行插值
                    spline = interp1d(x_unique, y_unique, kind='cubic')
                    # 生成更平滑的数据点
                    xnew = np.linspace(x_unique.min(), x_unique.max(), 300)
                    ynew = spline(xnew)
                    # 绘图
                    ax.plot(xnew, ynew, marker='', label=f'Error Rate = {err_rate * 100:.0f}%')  # 不显示标记
                else:
                    # 如果点数不足以进行插值，直接绘制原始数据
                    ax.plot(x_unique, y_unique, 'o', linestyle='-',
                            label=f'Error Rate = {err_rate * 100:.0f}%')
            else:
                # 如果点数不足以进行插值，直接绘制原始数据
                ax.plot(subset['Correct Rate'], subset[y_name], 'o', linestyle='-',
                        label=f'Error Rate = {err_rate * 100:.0f}%')
        # # 绘图设置
        # fig, ax = plt.subplots(figsize=(10, 6))
        #
        # # 不同的错误率
        # error_rates = df['Error Rate'].unique()
        # for err_rate in error_rates:
        #     subset = df[df['Error Rate'] == err_rate]
        #     if len(subset) > 3:  # 确保每个子集有足够的点
        #         # 数据点
        #         x = subset['Correct Rate']
        #         y = subset[y_name]
        #         # 使用滑动窗口平滑数据
        #         window_size = 3  # 滑动窗口大小设为3
        #         y_smoothed = y.rolling(window=window_size, center=True).mean()  # 计算移动平均，center=True表示窗口中心对齐当前值
        #         # 除去NaN值
        #         valid_indices = ~y_smoothed.isna()
        #         x = x[valid_indices]
        #         y_smoothed = y_smoothed[valid_indices]
        #         # 创建样条插值对象
        #         spline = interp1d(x, y_smoothed, kind='cubic')
        #         # 生成更平滑的数据点
        #         xnew = np.linspace(x.min(), x.max(), 300)
        #         ynew = spline(xnew)
        #         # 绘图
        #         ax.plot(xnew, ynew, marker='', label=f'Error Rate = {err_rate * 100:.0f}%')  # 不显示标记
        #         # 标记原始数据点
        #         # ax.plot(x, y_smoothed, 'o', label=f'Original Points at Error Rate = {err_rate * 100:.0f}%')
        #     else:
        #         # 如果点数不足以进行插值，直接绘制原始数据
        #         ax.plot(subset['Correct Rate'], subset[y_name], 'o', linestyle='-',
        #                 label=f'Error Rate = {err_rate * 100:.0f}%')

        ax.set_xlabel('Correct Rate')
        ax.set_ylabel(y_name)
        ax.set_title(f'{y_name} vs Correct Rate for Different Error Rates')
        ax.legend()
        plt.grid(True)
        plt.savefig(f'plot1_total_{base_file_name}_{y_name}_smooth.png')  # Save the first plot
        plt.show()

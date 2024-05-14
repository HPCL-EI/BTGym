


# 画出图
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

# Assuming the CSV data is saved in a file called 'data.csv'
# file_name = 'output_summary_bt_cys'
# file_name = 'output_summary_bt'
# file_name = 'EXP_2_output_summary_bt_data_small_100_bigerror'
# file_name = 'EXP_2_output_summary_bt_data_small_100_bigerror_0_1_5_10'
# file_name = 'EXP_2_output_summary_bt_data_small_100_bigerror_0_1_5_10_heuristic_choice=1'
# file_name = 'EXP_2_output_summary_bt_data_small_100_bigerror_0_5_1_3_5_heuristic_choice=1'
file_name = 'EXP_2_output_summary_bt_data_small_100_bigerror_heuristic=0_noRandom'

file_path = file_name+'.csv'
df = pd.read_csv(file_path)

y_name = "Total Current Cost"
# y_name = "Total Expanded Num"
# y_name = "Total Planning Time Total"


smooth = False

if not smooth:
    # Plotting
    # 无平滑处理
    fig, ax = plt.subplots(figsize=(10, 6))
    # Unique error rates to plot
    error_rates = df['Error Rate'].unique()
    for err_rate in error_rates:
        subset = df[df['Error Rate'] == err_rate]
        # if err_rate==1:
        #     continue
        ax.plot(subset['Correct Rate'], subset[y_name], marker='o', label=f'Error Rate = {err_rate*100}%')
    ax.set_xlabel('Correct Rate')
    ax.set_ylabel(y_name)
    ax.set_title(f'{y_name} vs Correct Rate for Different Error Rates')
    ax.legend()
    plt.grid(True)
    plt.savefig('plot1_total_'+file_name+'_'+y_name+'.png')  # Save the first plot
    plt.show()

else:

    # 绘图设置
    fig, ax = plt.subplots(figsize=(10, 6))

    # 不同的错误率
    error_rates = df['Error Rate'].unique()
    for err_rate in error_rates:
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
            # 创建样条插值对象
            spline = interp1d(x, y_smoothed, kind='cubic')
            # 生成更平滑的数据点
            xnew = np.linspace(x.min(), x.max(), 300)
            ynew = spline(xnew)
            # 绘图
            ax.plot(xnew, ynew, marker='', label=f'Error Rate = {err_rate * 100:.0f}%')  # 不显示标记
            # 标记原始数据点
            # ax.plot(x, y_smoothed, 'o', label=f'Original Points at Error Rate = {err_rate * 100:.0f}%')
        else:
            # 如果点数不足以进行插值，直接绘制原始数据
            ax.plot(subset['Correct Rate'], subset[y_name], 'o', linestyle='-',
                    label=f'Error Rate = {err_rate * 100:.0f}%')

    ax.set_xlabel('Correct Rate')
    ax.set_ylabel('Total Expanded Num')
    ax.set_title('Total Expanded Num vs Correct Rate for Different Error Rates')
    ax.legend()
    plt.grid(True)
    plt.savefig('plot1_total_' + file_name + '_' + y_name + '_smooth.png')  # Save the first plot
    plt.show()

    # # 绘图设置
    # fig, ax = plt.subplots(figsize=(10, 6))
    # # 不同的错误率
    # error_rates = df['Error Rate'].unique()
    # for err_rate in error_rates:
    #     subset = df[df['Error Rate'] == err_rate]
    #     if len(subset) > 3:  # 确保每个子集有足够的点进行三次样条插值
    #         # 数据点
    #         x = subset['Correct Rate']
    #         y = subset[y_name]
    #         # 创建样条插值对象
    #         spline = interp1d(x, y, kind='cubic')
    #         # 生成更平滑的数据点
    #         xnew = np.linspace(x.min(), x.max(), 300)
    #         ynew = spline(xnew)
    #         # 绘图
    #         ax.plot(xnew, ynew, marker='', label=f'Error Rate = {err_rate * 100:.0f}%')  # 不显示标记
    #     else:
    #         # 如果点数不足以进行插值，直接绘制原始数据
    #         ax.plot(subset['Correct Rate'], subset[y_name], marker='o', linestyle='-',
    #                 label=f'Error Rate = {err_rate * 100:.0f}%')
    # ax.set_xlabel('Correct Rate')
    # ax.set_ylabel('Total Expanded Num')
    # ax.set_title('Total Expanded Num vs Correct Rate for Different Error Rates')
    # ax.legend()
    # plt.grid(True)
    # plt.savefig('plot1_total_'+file_name+'_'+y_name+'_smooth.png')  # Save the first plot
    # plt.show()



# 全部都求平均画出来
# Group by 'Correct Rate' and calculate the mean of 'Total Expanded Num'
# grouped_data = df.groupby('Correct Rate')['Total Expanded Num'].mean().reset_index()
#
# # Plotting the curve
# plt.figure(figsize=(10, 6))
# plt.plot(grouped_data['Correct Rate'], grouped_data['Total Expanded Num'], marker='o')
# plt.xlabel('Correct Rate')
# plt.ylabel('Average Expanded Num')
# plt.title('Average Expanded Num vs Correct Rate')
# plt.grid(True)
# plt.savefig('plot2_average_expanded_num_'+file_name+'.png')  # Save the second plot
# plt.show()

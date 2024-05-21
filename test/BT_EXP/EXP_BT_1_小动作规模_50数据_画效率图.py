import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 导入数据
# file_path = 'average_results_data=50_size=70.csv'
file_path = 'EXP_1_average_results_100_size=90_20240512232528.csv'
average_df = pd.read_csv(file_path)

# 假设heuristic_choices是从数据中提取的独立值
heuristic_choices = average_df['Heuristic_Choice'].unique()
# 映射Heuristic Choice到描述性的标签
heuristic_labels = {
   -1: "No Heuristic",
    0: "Non-optimal Fast Heuristic",
    1: "Optimal Heuristic"
}

y_name = 'Planning_Time_Total'
# y_name = 'Expanded_Number'

smooth = True
if smooth:
    plt.figure(figsize=(10, 6))
    for heuristic_choice in heuristic_choices:
        subset = average_df[average_df['Heuristic_Choice'] == heuristic_choice]
        if not subset.empty:
            x = subset['Action_Space_Size']
            y = subset[y_name]
            # 使用滑动窗口平滑数据
            window_size = 5  # 滑动窗口大小设为3
            y_smoothed = y.rolling(window=window_size, center=True).mean()  # 计算移动平均，center=True表示窗口中心对齐当前值

            # 移除y_smoothed中的NaN值，调整x以匹配y_smoothed的长度
            valid_indices = y_smoothed.notna()  # 获取非NaN值的索引
            y_smoothed = y_smoothed[valid_indices]
            x_smoothed = x[valid_indices]

            plt.plot(x_smoothed, y_smoothed, label=heuristic_labels[heuristic_choice])

    plt.title('Average Planning Time Total vs Action Space Size')
    plt.xlabel('Action Space Size')
    plt.ylabel('Average '+y_name)
    plt.legend()
    plt.grid(True)

    # plt.figure(figsize=(10, 6))
    # for heuristic_choice in heuristic_choices:
    #     subset = average_df[average_df['Heuristic_Choice'] == heuristic_choice]
    #     if not subset.empty:
    #         x = subset['Action_Space_Size']
    #         y = subset[y_name]
    #         # 使用多项式拟合
    #         polynomial_coefficients = np.polyfit(x, y, deg=min(len(x)-1, 3))  # 使用较低的多项式阶数，最多3阶
    #         poly_function = np.poly1d(polynomial_coefficients)
    #         xnew = np.linspace(x.min(), x.max(), 300)  # 创建更细的x值用于平滑
    #         ynew = poly_function(xnew)  # 计算新的y值
    #         plt.plot(xnew, ynew, label=heuristic_labels[heuristic_choice])
    #
    # plt.title('Average Planning Time Total vs Action Space Size')
    # plt.xlabel('Action Space Size')
    # plt.ylabel('Average '+y_name)
    # plt.legend()
    # plt.grid(True)
else:
    for heuristic_choice in heuristic_labels.keys():
        subset = average_df[average_df['Heuristic_Choice'] == heuristic_choice]
        if not subset.empty:
            x = subset['Action_Space_Size']
            y = subset[y_name]
            plt.plot(x, y, marker='o', linestyle='-', label=heuristic_labels[heuristic_choice])  # 绘制点和线

    plt.title('Average Planning Time Total vs Action Space Size')
    plt.xlabel('Action Space Size')
    plt.ylabel('Average ' + y_name)
    plt.legend()
    plt.grid(True)

# 保存图像的名称需要定义时间字符串
from datetime import datetime
time_str = datetime.now().strftime('%Y%m%d%H%M%S')
if smooth:
    plt.savefig(f'EXP_1_smooth_average_{y_name}_plot_data=100_size=90.png')
else:
    plt.savefig(f'EXP_1_average_{y_name}_plot_data=100_size=90.png')

plt.show()





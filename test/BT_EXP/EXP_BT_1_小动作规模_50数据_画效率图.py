import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 导入数据
file_path = 'average_results_data=50_size=70.csv'
average_df = pd.read_csv(file_path)

# 假设heuristic_choices是从数据中提取的独立值
heuristic_choices = average_df['Heuristic_Choice'].unique()
# 映射Heuristic Choice到描述性的标签
heuristic_labels = {
    -1: "No Heuristic",
    0: "Non-optimal Fast Heuristic",
    1: "Optimal Heuristic"
}

plt.figure(figsize=(10, 6))

for heuristic_choice in heuristic_choices:
    subset = average_df[average_df['Heuristic_Choice'] == heuristic_choice]
    if not subset.empty:
        x = subset['Action_Space_Size']
        y = subset['Planning_Time_Total']
        # 使用多项式拟合
        polynomial_coefficients = np.polyfit(x, y, deg=min(len(x)-1, 3))  # 使用较低的多项式阶数，最多3阶
        poly_function = np.poly1d(polynomial_coefficients)
        xnew = np.linspace(x.min(), x.max(), 300)  # 创建更细的x值用于平滑
        ynew = poly_function(xnew)  # 计算新的y值
        plt.plot(xnew, ynew, label=heuristic_labels[heuristic_choice])

plt.title('Average Planning Time Total vs Action Space Size')
plt.xlabel('Action Space Size')
plt.ylabel('Average Planning Time Total')
plt.legend()
plt.grid(True)

# for heuristic_choice in heuristic_labels.keys():
#     subset = average_df[average_df['Heuristic_Choice'] == heuristic_choice]
#     if not subset.empty:
#         x = subset['Action_Space_Size']
#         y = subset['Planning_Time_Total']
#         plt.plot(x, y, marker='o', linestyle='-', label=heuristic_labels[heuristic_choice])  # 绘制点和线
#
# plt.title('Average Planning Time Total vs Action Space Size')
# plt.xlabel('Action Space Size')
# plt.ylabel('Average Planning Time Total')
# plt.legend()
# plt.grid(True)

# 保存图像的名称需要定义时间字符串
from datetime import datetime
time_str = datetime.now().strftime('%Y%m%d%H%M%S')
plt.savefig(f'average_planning_time_plot_data=50_size=70.png')

plt.show()





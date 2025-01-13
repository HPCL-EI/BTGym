import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from btgym.utils import ROOT_PATH
import pandas as pd

scene = "RW"
difficulty="multi"
maxep = 20


# Load the CSV file
file_path = f'{ROOT_PATH}/../z_benchmark/output_algo_act_num/{scene}_{difficulty}_maxep={maxep}_act_num.csv'
data = pd.read_csv(file_path)


A = data['opt_h0'].dropna().tolist()
B = data['opt_h0_llm'].dropna().tolist()
C = data['obtea'].dropna().tolist()
D = data['bfs'].dropna().tolist()


# 统计元素频次
counter1 = Counter(A)
counter2 = Counter(B)
counter3 = Counter(C)
counter4 = Counter(D)

# 合并四个Counter对象，以便我们可以获得完整的x轴范围
all_counts = counter1 + counter2 + counter3 + counter4
sorted_x = sorted(all_counts.keys())

# 设置柱状图的宽度和位置
bar_width = 0.15
x1 = range(len(sorted_x))
x2 = [x + bar_width for x in x1]
x3 = [x + bar_width for x in x2]
x4 = [x + bar_width for x in x3]

# 初始化y值为0，用于累加柱状图高度（如果需要重叠）
y1 = [counter1[x] if x in counter1 else 0 for x in sorted_x]
y2 = [counter2[x] if x in counter2 else 0 for x in sorted_x]
y3 = [counter3[x] if x in counter3 else 0 for x in sorted_x]
y4 = [counter4[x] if x in counter4 else 0 for x in sorted_x]


alpha = 0.20
# 绘制柱状图
plt.bar(x4, y4, width=bar_width, label='BT Expansion', color='blue', alpha=alpha)

plt.bar(x3, y3, width=bar_width, label='OBTEA', color='green', alpha=alpha)

plt.bar(x2, y2, width=bar_width, label='HOBTEA', color='orange', alpha=alpha)

plt.bar(x1, y1, width=bar_width, label='HOBTEA-Oracle', color='red', alpha=alpha)



# 设置x轴标签
plt.xticks([x + bar_width for x in range(len(sorted_x))], sorted_x)

# 添加标题和轴标签
plt.title(f'{scene}_{difficulty}_maxep={maxep}')
plt.xlabel('Region Distance')
plt.ylabel('Frequency')

# 添加图例
plt.legend()
plt.subplots_adjust(bottom=0.3)
plt.savefig(f'{ROOT_PATH}/../z_benchmark/output_algo_act_num/{scene}_{difficulty}_maxep={maxep}.png',
            dpi=100, bbox_inches='tight')
# 显示图形
plt.show()



# A = algo_act_num_ls['opt_h0']
# B = algo_act_num_ls['opt_h0_llm']
# C = algo_act_num_ls['obtea']
# D = algo_act_num_ls['bfs']

# # 使用Counter统计频次
# counts1 = Counter(A)
# counts2 = Counter(B)
# counts3 = Counter(C)
# counts4 = Counter(D)
#
# # 合并四个Counter对象，以便我们可以获得完整的x轴范围
# all_counts = counts1 + counts2 + counts3 + counts4
# sorted_x = sorted(all_counts.keys())
#
#
# # 初始化y值为0，用于累加柱状图高度（如果需要重叠）
# y1 = [counts1[x] if x in counts1 else 0 for x in sorted_x]
# y2 = [counts2[x] if x in counts2 else 0 for x in sorted_x]
# y3 = [counts3[x] if x in counts3 else 0 for x in sorted_x]
# y4 = [counts4[x] if x in counts4 else 0 for x in sorted_x]
#
# # 绘制第一个柱状图
# y_bottom = 0
#
# # BTExpansion
# plt.bar(sorted_x, y4, width=0.8, color='green', edgecolor='black', label='BT Expansion', alpha=0.1)
# poly = np.polyfit(sorted_x, y4, deg=2)
# y_value = np.polyval(poly, sorted_x)
# plt.plot(sorted_x, y_value, color='green', alpha=0.1)
# plt.ylim(bottom=y_bottom)
#
# # OBTEA
# plt.bar(sorted_x, y3, width=0.8, color='lightblue', edgecolor='black', label='OBTEA', alpha=0.2)
# poly = np.polyfit(sorted_x, y3, deg=2)
# y_value = np.polyval(poly, sorted_x)
# plt.plot(sorted_x, y_value, color='lightblue', alpha=0.2)
#
# # HOBTEA
# # 绘制第二个柱状图（相邻但不重叠）
# # 注意：如果你想要重叠，可以将bottom参数设置为y1，但通常不推荐这样做
# plt.bar(sorted_x, y2, width=0.8, color='orange', edgecolor='black', label='HOBTEA', alpha=0.1)
# poly = np.polyfit(sorted_x, y2, deg=2)
# y_value = np.polyval(poly, sorted_x)
# plt.plot(sorted_x, y_value, color='orange', alpha=0.1)
#
# # HOBTEA - Oracle
# plt.bar(sorted_x, y1, width=0.8, color='red', edgecolor='black', label='HOBTEA-Oracle', alpha=0.1)
# # 绘制曲线
# poly = np.polyfit(sorted_x, y1, deg=2)
# y_value = np.polyval(poly, sorted_x)
# plt.plot(sorted_x, y_value, color='red', alpha=0.1)
#
#
# # 添加标题和轴标签
# plt.title(f'{scene}_{difficulty}_maxep={maxep}')
# plt.xlabel('Values')
# plt.ylabel('Frequency')
#
# # 添加图例
# plt.legend()
# plt.subplots_adjust(bottom=0.3)
# plt.savefig(f'./算法效率的对比图/{scene}_{difficulty}_maxep={maxep}.png',
#             dpi=100, bbox_inches='tight')
# # 显示图形
# plt.show()

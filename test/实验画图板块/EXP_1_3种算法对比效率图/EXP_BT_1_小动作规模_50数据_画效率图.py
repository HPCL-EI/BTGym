import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
# 尝试设置数学文本的字体，但这可能不会完全奏效
matplotlib.rcParams['mathtext.fontset'] = 'stix'  # STIX 字体风格更接近 Times New Roman
from matplotlib.ticker import MultipleLocator
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

font1 = {'family': 'Times New Roman','color': 'Black','weight': 'bold','size': 48} #normal
font2 = {'family': 'Times New Roman','size': 34,'weight': 'bold'}
# font3 = {'family': 'Times New Roman','color': 'Black','weight': 'bold','size': 38}
from matplotlib.ticker import MultipleLocator, FuncFormatter

# 导入数据
file_path = 'EXP_1_average_results_100__size=90_4h.csv'
average_df = pd.read_csv(file_path)

# 假设heuristic_choices是从数据中提取的独立值
heuristic_choices = average_df['Heuristic_Choice'].unique()
# 映射Heuristic Choice到描述性的标签
# heuristic_labels = {
#    -1: "OBTEA",
#    -2: "BT-Expansion",
#     0: "Fast Heuristic",
#     1: "Optimal Heuristic"
# }
heuristic_labels = {
   -1: "OBTEA",
   -2: "BT-Expansion",
    0: "HBTP-F",
    1: "HBTP-O"
}


# 重新排列 heuristic_choices 以符合图例的期望顺序
ordered_heuristic_choices = [-2, -1, 1, 0]

# 配色方案
colors = {
    1: "#d62728",   # 红色 "OBTEA"
    0: "#2ca02c",  # 橙色 "BT-Expansion"
    -2:  "#1f77b4",    # 绿色"Fast Heuristic"
    -1:  "#ff7f0e",  # 蓝色"Optimal Heuristic"
}

y_name = 'Planning_Time_Total'

smooth = True
if smooth:
    fig, ax = plt.subplots(figsize=(10, 6))
    for heuristic_choice in ordered_heuristic_choices:
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

            ax.plot(x_smoothed, y_smoothed, label=heuristic_labels[heuristic_choice], color=colors[heuristic_choice], linewidth=3)

    import numpy as np

    plt.yticks(np.arange(0, 0.8, 0.2))  # 从 0 到 1，以 0.2 为步长

    plt.xlabel('Action Space Size',fontdict=font1)
    label = plt.ylabel('Planning Time(s)',fontdict=font1)
    label.set_position((label.get_position()[0], label.get_position()[1] - 0.1))  # 调整 y 轴标签的 y 位置

    ax.legend(prop=font2, loc='upper left')
    plt.grid(True)

    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(48) for label in labels]

    # 调整布局以防止标签被截断
    plt.tight_layout()
    plt.savefig(f"EXP_BT_1_time_vs_action_space.pdf", dpi=50, bbox_inches='tight', format='pdf')
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
# 尝试设置数学文本的字体，但这可能不会完全奏效
matplotlib.rcParams['mathtext.fontset'] = 'stix'  # STIX 字体风格更接近 Times New Roman
from matplotlib.ticker import MultipleLocator
font1 = {'family': 'Times New Roman','color': 'Black','weight': 'normal','size': 32}
font2 = {'family': 'Times New Roman','size': 22}
font3 = {'family': 'Times New Roman','color': 'Black','weight': 'normal','size': 38}
from matplotlib.ticker import MultipleLocator, FuncFormatter
heuristic=0
# 文件名和路径
base_file_name = f'EXP_2_output_summary_bt_data_small_100_bigerror_heuristic={heuristic}_seed='
file_path = "exp_output_100/"
seeds = range(0, 50)
# 是否平滑
smooth = True

window_size = 1  # 窗口大小
error_scale = 1  # 错误阴影缩放变量


# 50-65
# 不透明度

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
# y_names = ["Total Current Cost"]
y_names = ["Total Expanded Num"]

# y_names = ["Total Current Cost", "Total Expanded Num", "Total Planning Time Total"]
# y_names = ["Total Current Cost", "Total Expanded Num"]

def smooth_data(data, window_size):
    return data.rolling(window=window_size, min_periods=1).mean()

for y_name in y_names:
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    # Unique error rates to plot
    error_rates = df['Error Rate'].unique()
    for err_rate in error_rates:
        if err_rate == 0.5:
            continue
        subset = df[df['Error Rate'] == err_rate]
        # 计算平均值和标准误差
        mean_data = subset.groupby('Correct Rate')[y_name].mean().reset_index()
        std_data = subset.groupby('Correct Rate')[y_name].std().reset_index()

        if smooth:
            mean_data[y_name] = smooth_data(mean_data[y_name], window_size)
            std_data[y_name] = smooth_data(std_data[y_name], window_size)

        # 绘制曲线和误差填充
        ax.plot(mean_data['Correct Rate'], mean_data[y_name], marker='o', \
                label=f'Error Rate = {int(err_rate*100)}%',linewidth=3)
        ax.fill_between(mean_data['Correct Rate'], mean_data[y_name] - error_scale * std_data[y_name],
                        mean_data[y_name] + error_scale * std_data[y_name], alpha=0.05)

    ax.set_xlabel('Correct Rate',fontdict=font1)
    ax.set_ylabel(y_name,fontdict=font1)
    # ax.set_title(f'{y_name} vs Correct Rate for Different Error Rates')
    ax.legend(prop=font2)
    plt.grid(True)

    # 设置 x 轴刻度和标签格式
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x * 100)}%'))


    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(30) for label in labels]

    # 调整布局以防止标签被截断
    plt.tight_layout()
    plt.savefig(f"EXP_BT_2_plot_error_h{heuristic}.pdf", dpi=100, bbox_inches='tight', format='pdf')
    plt.show()

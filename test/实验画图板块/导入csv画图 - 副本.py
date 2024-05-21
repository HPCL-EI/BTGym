import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
# 尝试设置数学文本的字体，但这可能不会完全奏效
matplotlib.rcParams['mathtext.fontset'] = 'stix'  # STIX 字体风格更接近 Times New Roman
from matplotlib.ticker import MultipleLocator
font1 = {'family': 'Times New Roman','color': 'Black','weight': 'normal','size': 32}
font2 = {'family': 'Times New Roman','size': 30}
font3 = {'family': 'Times New Roman','color': 'Black','weight': 'normal','size': 38}
from matplotlib.ticker import MultipleLocator, FuncFormatter
# Define the metrics
metrics = [
    'Success_rate',
    'Average_distance',
    'Expanded_num',
    'Act_space'
]

metric2label={
    'Success_rate': 'Success Rate',
    'Average_distance': 'Nearest Neighbor Distance',
    'Expanded_num': 'Expanded Num',
    'Act_space': 'Action Space Size'
}

name2label={
    'easy':'Easy',
    "medium":'Medium',
    "hard":"Hard"
}

# Set the directory where the CSV files are stored
results_dir = './'

# Plotting each metric from its corresponding CSV file
for metric in metrics:
    # Construct the file path

    if metric=="Success_rate" or metric=="Act_space" or metric =="Expanded_num":
        file_path = f'{results_dir}smoothed_{metric.replace(" ", "_")}_mean_std_xxx.csv'
    else:
        file_path = f'{results_dir}smoothed_{metric.replace(" ", "_")}_mean_std.csv'

    # Read the CSV file
    df = pd.read_csv(file_path)
    # Scale rounds by 10
    df['round'] = df['round'] * 10
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each heuristic's mean and fill the area between mean ± std
    for name in ['easy', 'medium', 'hard']:
        ax.plot(df['round'], df[f'{name}_mean'], linewidth=2,\
                label=name2label[name],markersize=8, marker='o')  # Add marker='o' to show data points
        ax.fill_between(df['round'], df[f'{name}_mean'] - df[f'{name}_std'], df[f'{name}_mean'] + df[f'{name}_std'],
                         alpha=0.1)

    # plt.title(f'{metric} over Rounds (Smoothed)')
    ax.set_xlabel('Number of Training Samples',fontdict=font1)
    ax.set_ylabel(metric2label[metric],fontdict=font1)
    plt.grid(True)
    ax.legend(prop=font2)

    # Adjust the y-axis for Success Rate to show as percentage
    if metric == 'Success_rate':
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax.set_yticks([0.2, 0.4, 0.6,0.8])  # Set specific ticks at 20%, 40%, and 100%


    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(30) for label in labels]

    # 调整布局以防止标签被截断
    plt.tight_layout()
    plt.savefig(f'EXP_LLM_{metric.replace(" ", "_")}_over_Rounds.pdf', dpi=100, bbox_inches='tight', format='pdf')
    plt.show()

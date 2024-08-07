import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
# 尝试设置数学文本的字体，但这可能不会完全奏效
matplotlib.rcParams['mathtext.fontset'] = 'stix'  # STIX 字体风格更接近 Times New Roman
from matplotlib.ticker import MultipleLocator
font1 = {'family': 'Times New Roman','color': 'Black','weight': 'normal','size': 32}
font2 = {'family': 'Times New Roman','size': 24}
font3 = {'family': 'Times New Roman','color': 'Black','weight': 'normal','size': 38}
from matplotlib.ticker import MultipleLocator, FuncFormatter
# Define the metrics
metrics = [
    'Test Success Rate Once',
    'Average Distance',
    'Average Expanded Num',
    'Average Act Space'
]

metric2label={
    'Test Success Rate Once': 'Success Rate',
    'Average Distance': 'Nearest Neighbor Distance',
    'Average Expanded Num': 'Expanded Num',
    'Average Act Space': 'Action Space Size'
}

# Set the directory where the CSV files are stored
results_dir = './'

# Plotting each metric from its corresponding CSV file
for metric in metrics:
    # Construct the file path
    file_path = f'{results_dir}{metric.replace(" ", "_")}_mean_std.csv'

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each heuristic's mean and fill the area between mean ± std
    for name in ['easy', 'medium', 'hard']:
        ax.plot(df['Round'], df[f'{name} Mean'], label=name, marker='o')  # Add marker='o' to show data points
        ax.fill_between(df['Round'], df[f'{name} Mean'] - df[f'{name} Std'], df[f'{name} Mean'] + df[f'{name} Std'],
                         alpha=0.08)

    # plt.title(f'{metric} over Rounds (Smoothed)')
    ax.set_xlabel('Number of Training Samples',fontdict=font1)
    ax.set_ylabel(metric2label[metric],fontdict=font1)
    plt.grid(True)
    ax.legend(prop=font2)

    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(30) for label in labels]

    # 调整布局以防止标签被截断
    plt.tight_layout()
    plt.savefig(f'{metric.replace(" ", "_")}_over_Rounds_with_error_fill_smoothed.png.pdf', dpi=100, bbox_inches='tight', format='pdf')
    plt.show()

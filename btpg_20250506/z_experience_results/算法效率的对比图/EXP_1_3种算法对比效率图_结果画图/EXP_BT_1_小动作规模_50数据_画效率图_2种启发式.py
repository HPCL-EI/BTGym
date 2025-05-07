import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Set global font to Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'

font1 = {'family': 'Times New Roman', 'color': 'Black', 'weight': 'bold', 'size': 40}
font2 = {'family': 'Times New Roman', 'size': 32, 'weight': 'bold'}
font3 = {'family': 'Times New Roman', 'color': 'Black', 'weight': 'bold', 'size': 40}

# Load data from CSV
file_path = 'algorithm_comparison_results.csv'
df = pd.read_csv(file_path)

# Plotting parameters
colors = {
    'H1 Avg Planning Time': "#d62728",
    'H0 Avg Planning Time': "#2ca02c"  # Orange
}
h_labels = {
    'H0 Avg Planning Time': "Fast Heuristic",
    'H1 Avg Planning Time': "Optimal Heuristic"
}

window_size = 5

fig, ax = plt.subplots(figsize=(10, 6))

for column in ['H1 Avg Planning Time', 'H0 Avg Planning Time']:
    x = df['Action Length']
    y = df[column]

    # Smooth data using a rolling window
    y_smoothed = y.rolling(window=window_size, center=True).mean()

    # Remove NaN values and adjust x accordingly
    valid_indices = y_smoothed.notna()
    y_smoothed = y_smoothed[valid_indices]
    x_smoothed = x[valid_indices]

    ax.plot(x_smoothed, y_smoothed, label=h_labels[column], color=colors[column], linewidth=5)
    # ax.plot(x_smoothed, y_smoothed, label=column, linewidth=5)

plt.xlabel('Optimal Path Length', fontdict=font1)
plt.ylabel('Planning Time (s)', fontdict=font1)
ax.legend(prop=font2)
plt.grid(True)

labels = ax.get_xticklabels() + ax.get_yticklabels()
for label in labels:
    label.set_fontname('Times New Roman')
    label.set_fontsize(40)

plt.tight_layout()
plt.savefig("EXP_BT_1_time_vs_act_len.pdf", dpi=100, bbox_inches='tight', format='pdf')
plt.show()

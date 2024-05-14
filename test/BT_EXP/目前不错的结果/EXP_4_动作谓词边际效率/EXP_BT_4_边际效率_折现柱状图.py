import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_data(csv_path, bar_orientation='vertical', font_size=12, top_n=10, save_path='plot.png',label_rotation=45):
    # Load data
    data = pd.read_csv(csv_path)
    data.set_index(data.columns[0], inplace=True)  # Set the first column as index if not already

    # Determine the number of entries to display
    if top_n is None or top_n > len(data):
        top_n = len(data)  # Use the total number of entries if top_n is None or too large
    data = data.head(top_n)  # Filter to show only the top_n entries

    print(f"Displaying top {top_n} entries out of {len(data)} available.")
    # Plot settings
    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid")

    # Determine orientation
    if bar_orientation == 'horizontal':
        ax = sns.barplot(x=data.columns[0], y=data.index, data=data)
    else:
        ax = sns.barplot(x=data.index, y=data.columns[0], data=data)

    # Add labels to each bar
    for p in ax.patches:
        if bar_orientation == 'horizontal':
            ax.annotate(f'{p.get_width():.1f}',
                        (p.get_width(), p.get_y() + p.get_height() / 2),
                        ha='left', va='center', fontsize=font_size, color='black', xytext=(5, 0),
                        textcoords='offset points')
            ax.scatter(p.get_width(), p.get_y() + p.get_height() / 2, color='red')
        else:
            ax.annotate(f'{p.get_height():.1f}',
                        (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha='center', va='bottom', fontsize=font_size, color='black', xytext=(0, 5),
                        textcoords='offset points')
            ax.scatter(p.get_x() + p.get_width() / 2, p.get_height(), color='red')

    # Connect points with a line plot
    if bar_orientation == 'horizontal':
        xs = [p.get_width() for p in ax.patches]
        ys = [p.get_y() + p.get_height() / 2 for p in ax.patches]
    else:
        xs = [p.get_x() + p.get_width() / 2 for p in ax.patches]
        ys = [p.get_height() for p in ax.patches]
    plt.plot(xs, ys, linestyle='-', marker='o', color='blue')

    # Adjust plot and show
    plt.title('Impact of Predicates on Expanded Number Difference', fontsize=font_size + 2)
    plt.xlabel(data.columns[0] if bar_orientation == 'vertical' else 'Predicates', fontsize=font_size)
    plt.ylabel('Predicates' if bar_orientation == 'vertical' else data.columns[0], fontsize=font_size)

    # Rotate x-axis labels if vertical
    if bar_orientation == 'vertical':
        plt.xticks(rotation=label_rotation, fontsize=font_size)  # Rotate labels
    else:
        plt.yticks(fontsize=font_size)

    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()  # Adjust layout to make room for labels
    plt.savefig(save_path)  # Save the figure
    plt.show()

file_name = "EXP_BT_4_predicate_effects.csv"
# Example usage
plot_data(file_name, bar_orientation='vertical', font_size=15, top_n=13,
          save_path='EXP_BT_4_top_5_predicates.png',label_rotation=45) #top_n=None
# plot_data(file_name, bar_orientation='horizontal', font_size=12, top_n=13,
#           save_path='EXP_BT_4_top_5_predicates.png')

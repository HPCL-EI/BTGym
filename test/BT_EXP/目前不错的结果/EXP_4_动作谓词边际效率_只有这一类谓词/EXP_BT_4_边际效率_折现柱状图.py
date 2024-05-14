import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_data(csv_path, metric, bar_orientation='vertical', font_size=12, top_n=10, save_path='plot.png', label_rotation=45, plot_error=False):
    # Load data
    data = pd.read_csv(csv_path)
    data.set_index(data.columns[0], inplace=True)  # Set the first column as index if not already

    # Ensure the metric exists in the data
    if metric not in data.columns:
        raise ValueError(f"Metric '{metric}' not found in the data columns: {data.columns.tolist()}")

    # Sort data by the selected metric in descending order
    data = data.sort_values(by=metric, ascending=False)

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
        ax = sns.barplot(x=metric, y=data.index, data=data)
    else:
        ax = sns.barplot(x=data.index, y=metric, data=data)

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

    # Add error bars if requested
    if plot_error:
        error_metric = metric + '_std'
        if error_metric in data.columns:
            errors = data[error_metric].values
            if bar_orientation == 'horizontal':
                plt.errorbar(data[metric], data.index, xerr=errors, fmt='none', c='blue', capsize=5)
            else:
                plt.errorbar(data.index, data[metric], yerr=errors, fmt='none', c='blue', capsize=5)

    # Adjust plot and show
    plt.title(f'Impact of Predicates on {metric.replace("_", " ").title()}', fontsize=font_size + 2)
    plt.xlabel(metric.replace("_", " ").title() if bar_orientation == 'vertical' else 'Predicates', fontsize=font_size)
    plt.ylabel('Predicates' if bar_orientation == 'vertical' else metric.replace("_", " ").title(), fontsize=font_size)

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

file_name = "EXP_BT_4_summary_predicate_effects_only1.csv"
# Example usage
# expanded_num_diff
plot_data(file_name, metric='expanded_num_diff', bar_orientation='horizontal', font_size=15, top_n=None,
          save_path='EXP_BT_4_top_predicates.png', label_rotation=-45, plot_error=True)
# plot_data(file_name, metric='time_diff', bar_orientation='vertical', font_size=15, top_n=None, save_path='EXP_BT_4_top_time_diff.png', label_rotation=45)
# plot_data(file_name, metric='cost_diff', bar_orientation='horizontal', font_size=12, top_n=10, save_path='EXP_BT_4_top_cost_diff.png')

import pandas as pd
import matplotlib.pyplot as plt

names = ['easy', 'medium', 'hard']
metrics_files = [f'results/{name}_metrics_20240520_1.csv' for name in names]

# Reading data from the CSV files
data = {name: pd.read_csv(file) for name, file in zip(names, metrics_files)}

# Rounds scaled by 10
rounds = data['easy']['Round'] * 10
# Set the maximum rounds value
max_rounds = 100  # Example value, you can change it to your desired maximum round
# Filter the data based on max_rounds
filtered_data = {name: df[rounds <= max_rounds] for name, df in data.items()}
filtered_rounds = rounds[rounds <= max_rounds]


# Define the metrics to plot
metrics = [
    'Test Success Rate Once',
    'Average Distance',
    'Average Expanded Num',
    # 'Average Planning Time Total',
    # 'Average Act Space'
]

# Plotting
for metric in metrics:
    plt.figure(figsize=(10, 6))

    for name in names:
        plt.plot(filtered_rounds, filtered_data[name][metric], marker='o', label=name)

    plt.title(f'{metric} over Rounds')
    plt.xlabel('Round (x10)')
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend()
    plt.savefig(f'results/{metric.replace(" ", "_")}_over_Rounds.png')
    plt.show()

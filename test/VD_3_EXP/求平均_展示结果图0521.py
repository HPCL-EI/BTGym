import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def moving_average(data, window_size):
    """Calculate the moving average with a specified window size."""
    return data.rolling(window=window_size, min_periods=1, center=True).mean()

names = ['easy', 'medium', 'hard']

# Initialize dictionary to hold all data
all_data = {name: [] for name in names}

# Reading data from the CSV files for each run
for name in names:
    for num in range(3):
        if name == "easy":
            file_path = f'output_{name}_{num}/{name}_metrics_20240521.csv'
        else:
            file_path = f'output_{name}_{num}/{name}_metrics_20240520.csv'
        print(f"Checking file path: {file_path}")
        if os.path.exists(file_path):
            all_data[name].append(pd.read_csv(file_path))
        else:
            print(f"File {file_path} not found.")

# Calculate the mean and standard deviation for each metric
mean_data = {name: pd.concat(dfs).groupby('Round').mean().reset_index() for name, dfs in all_data.items() if dfs}
std_data = {name: pd.concat(dfs).groupby('Round').std().reset_index() for name, dfs in all_data.items() if dfs}

# Check if mean_data and std_data have any content
if not mean_data or not std_data:
    print("No data to process.")
else:
    # Rounds scaled by 10
    rounds = mean_data['easy']['Round'] * 10
    # Set the maximum rounds value
    max_rounds = 200  # Example value, you can change it to your desired maximum round

    # Filter the data based on max_rounds
    filtered_mean_data = {name: df[rounds <= max_rounds] for name, df in mean_data.items()}
    filtered_std_data = {name: df[rounds <= max_rounds] for name, df in std_data.items()}
    filtered_rounds = rounds[rounds <= max_rounds]

    # Define the metrics to plot
    metrics = [
        'Test Success Rate Once',
        'Average Distance',
        'Average Expanded Num',
        'Average Act Space'
    ]

    window_size = 10  # You can change this to the desired window size

    # Plotting with shaded error bars and smoothing
    for metric in metrics:
        plt.figure(figsize=(10, 6))

        for name in names:
            if name in filtered_mean_data and name in filtered_std_data:
                mean_values = filtered_mean_data[name][metric]
                std_values = filtered_std_data[name][metric]

                # Apply moving average
                smoothed_mean_values = moving_average(mean_values, window_size)
                smoothed_std_values = moving_average(std_values, window_size)

                plt.plot(filtered_rounds, smoothed_mean_values, marker='o', label=name)
                plt.fill_between(filtered_rounds, smoothed_mean_values - smoothed_std_values, smoothed_mean_values + smoothed_std_values, alpha=0.2)

        plt.title(f'{metric} over Rounds (Smoothed)')
        plt.xlabel('Round (x10)')
        plt.ylabel(metric)
        plt.grid(True)
        plt.legend()
        plt.savefig(f'results/{metric.replace(" ", "_")}_over_Rounds_with_error_fill_smoothed.png')
        plt.show()

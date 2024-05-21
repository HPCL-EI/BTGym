import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the difficulty levels and their corresponding directories
difficulties = ['easy', 'medium', 'hard']
base_dirs = [f'output_{difficulty}_{i}' for difficulty in difficulties for i in range(1, 4)]

# Initialize dictionaries to store aggregated data
mean_data = {difficulty: None for difficulty in difficulties}
std_data = {difficulty: None for difficulty in difficulties}

# Read and process data for each difficulty level
for difficulty in difficulties:
    all_dfs = []

    for i in range(1, 4):
        if difficulty == "easy":
            file_path = f'output_{difficulty}_{i}/{difficulty}_details_20240521.csv'
        else:
            file_path = f'output_{difficulty}_{i}/{difficulty}_details_20240520.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Calculate success rate for each round
            df['success_rate'] = df.groupby('round')['fail'].transform(lambda x: (x == 0).mean())
            # Group by 'round' and calculate mean
            grouped = df.groupby('round').agg({
                'act_space': 'mean',
                'expanded_num': 'mean',
                'average_distance': 'mean',  # Assuming this column exists in the CSV
                'success_rate': 'mean'       # Calculate mean success rate
            }).reset_index()
            all_dfs.append(grouped)
        else:
            print(f"File {file_path} not found.")

    if all_dfs:
        combined_df = pd.concat(all_dfs)
        mean_data[difficulty] = combined_df.groupby('round').mean().reset_index()
        std_data[difficulty] = combined_df.groupby('round').std().reset_index()

# Define a function to apply rolling window smoothing
def smooth_data(data, window_size=5):
    return data.rolling(window=window_size, min_periods=1, center=True).mean()

# Define the metrics to plot
metrics = ['act_space', 'expanded_num', 'average_distance', 'success_rate']

# Initialize dictionaries to store smoothed data
smoothed_mean_data = {difficulty: None for difficulty in difficulties}
smoothed_std_data = {difficulty: None for difficulty in difficulties}

# Apply smoothing and prepare data for saving
for difficulty in difficulties:
    if mean_data[difficulty] is not None:
        smoothed_mean_data[difficulty] = mean_data[difficulty].copy()
        smoothed_std_data[difficulty] = std_data[difficulty].copy()
        for metric in metrics:
            smoothed_mean_data[difficulty][metric] = smooth_data(mean_data[difficulty][metric])
            smoothed_std_data[difficulty][metric] = smooth_data(std_data[difficulty][metric])

# Combine smoothed data and save to CSV files
for metric in metrics:
    combined_df = pd.DataFrame({'round': smoothed_mean_data[difficulties[0]]['round']})
    for difficulty in difficulties:
        combined_df[f'{difficulty}_mean'] = smoothed_mean_data[difficulty][metric]
        combined_df[f'{difficulty}_std'] = smoothed_std_data[difficulty][metric]

    combined_df.to_csv(f'smoothed_{metric.replace(" ", "_").capitalize()}_mean_std.csv', index=False)

# Plotting each metric
for metric in metrics:
    plt.figure(figsize=(10, 6))

    for difficulty in difficulties:
        if smoothed_mean_data[difficulty] is not None:
            rounds = smoothed_mean_data[difficulty]['round']
            means = smoothed_mean_data[difficulty][metric]
            stds = smoothed_std_data[difficulty][metric]

            plt.plot(rounds, means, marker='o', label=difficulty)
            plt.fill_between(rounds, means - stds, means + stds, alpha=0.2)

    plt.title(f'{metric.replace("_", " ").capitalize()} over Rounds (Smoothed)')
    plt.xlabel('Round')
    plt.ylabel(metric.replace("_", " ").capitalize())
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{metric}_over_rounds_with_error_fill_smoothed.png')
    plt.show()

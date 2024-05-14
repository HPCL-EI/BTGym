import pandas as pd

# Load the data from the CSV file
# file_path = 'LLM_40.csv'

# file_path = 'LLM_40_last.csv'
# file_path = 'LLM_40_gpt3_Noreflect.csv'
file_path ='LLM_40_gpt3_reflect.csv'
data = pd.read_csv(file_path)


# Assuming there are exactly 40 entries and we need to split them into 4 groups of 10 each
data['Group'] = (data.index // 10) + 1
# data['Group'] = (data.index // 2) + 1

# Initialize a dictionary to store results
results = {}


# Define function to calculate group statistics
def calculate_group_statistics(group_data):
    # Calculate success rate: proportion of entries with err == 0
    success_rate = (group_data['err'] == 0).mean()
    # Calculate timeout rate: proportion of entries with Timeout == 1
    timeout_rate = (group_data['Timeout'] == 1).mean()
    # Calculate average time (total)
    average_time_total = group_data['time'].mean()
    # Calculate average time on successful completions
    successful_times = group_data[group_data['err'] == 0]['time']
    average_time_success = successful_times.mean() if not successful_times.empty else 0
    # Calculate average expanded number (total and success)
    expanded_num_total = group_data['exp'].mean()
    successful_expansions = group_data[group_data['err'] == 0]['exp']
    expanded_num_success = successful_expansions.mean() if not successful_expansions.empty else 0
    # Calculate cost for successful completions
    successful_costs = group_data[group_data['err'] == 0]['cost']
    cost_success = successful_costs.mean() if not successful_costs.empty else 0

    return {
        'Success Rate': success_rate,
        'Timeout Rate': timeout_rate,
        'Time (Total)': average_time_total,
        'Time (Success)': average_time_success,
        'Expanded Num (Total)': expanded_num_total,
        'Expanded Num (Success)': expanded_num_success,
        'Cost (Success)': cost_success
    }


# Calculate statistics for each group and overall
for i in range(1, 5):
    results[f'Group{i}'] = calculate_group_statistics(data[data['Group'] == i])

# Calculate overall statistics
results['Overall'] = calculate_group_statistics(data)

# Convert results to DataFrame for nicer display
results_df = pd.DataFrame(results).T

# Round numeric columns to 4 decimal places in the DataFrame
for col in results_df.columns:
    if results_df[col].dtype in ['float64', 'float32']:  # Check if the column is a float type
        results_df[col] = results_df[col].round(4)


# 打印完整
# Configure Pandas to display all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# Print results
print(results_df)
print("------------------------------")
print(results_df.to_csv(sep='\t', index=True))


# 写入文件
# Specify the file path for the output CSV
output_csv_path = f'Group_Statistics_{file_path}.csv'
# Write results to CSV
results_df.to_csv(output_csv_path, index=True)
# Optionally print the file path of the created CSV file
print(f'Results have been saved to: {output_csv_path}')

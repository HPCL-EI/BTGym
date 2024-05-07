

import pandas as pd

# Load the results from the previously saved CSV file
file_name = "gpt4_40_time=20240507 165325.csv"
# file_name = "history/llm_40_just_goal_gpt4.csv"
results_df = pd.read_csv(file_name)
results = results_df.to_dict(orient='records')  # Convert DataFrame back to list of dictionaries

# Check that the data is successfully loaded
print(f"Loaded {len(results)} results from '{file_name}'")



# 分别输出四类数据的准确率
# Initialize accuracy statistics by group
num_groups = 4
group_size = len(results) // num_groups
group_stats = []

# Calculate accuracy statistics for each group
for group_num in range(num_groups):
    group = results_df[group_num * group_size: (group_num + 1) * group_size]
    group_name = f"Group {group_num + 1}"

    act_acc = group['Act_Acc'].mean()
    pred_acc = group['Pred_Acc'].mean()
    obj_acc = group['Obj_Acc'].mean()

    group_stats.append({
        'Group': group_name,
        'Actions Accuracy': f"{act_acc:.3f}%",
        'Key Predicates Accuracy': f"{pred_acc:.3f}%",
        'Key Objects Accuracy': f"{obj_acc:.3f}%"
    })

# Calculate overall accuracies
overall_act_acc = results_df['Act_Acc'].mean()
overall_pred_acc = results_df['Pred_Acc'].mean()
overall_obj_acc = results_df['Obj_Acc'].mean()

# Append overall statistics to the group stats
group_stats.append({
    'Group': 'Overall',
    'Actions Accuracy': f"{overall_act_acc:.3f}%",
    'Key Predicates Accuracy': f"{overall_pred_acc:.3f}%",
    'Key Objects Accuracy': f"{overall_obj_acc:.3f}%"
})

# Convert group stats to a DataFrame and print as a table
group_stats_df = pd.DataFrame(group_stats)
print("\nAccuracy Statistics by Group:")
print(group_stats_df.to_string(index=False))

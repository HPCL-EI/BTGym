


# 画出图
import pandas as pd
import matplotlib.pyplot as plt


# Assuming the CSV data is saved in a file called 'data.csv'
file_path = 'output_summary_bt.csv'
df = pd.read_csv(file_path)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
# Unique error rates to plot
error_rates = df['Error Rate'].unique()

for err_rate in error_rates:
    subset = df[df['Error Rate'] == err_rate]
    ax.plot(subset['Correct Rate'], subset['Total Expanded Num'], marker='o', label=f'Error Rate = {err_rate}')

ax.set_xlabel('Correct Rate')
ax.set_ylabel('Total Expanded Num')
ax.set_title('Total Expanded Num vs Correct Rate for Different Error Rates')
ax.legend()
plt.grid(True)
plt.show()





# 全部都求平均画出来
# Group by 'Correct Rate' and calculate the mean of 'Total Expanded Num'
grouped_data = df.groupby('Correct Rate')['Total Expanded Num'].mean().reset_index()

# Plotting the curve
plt.figure(figsize=(10, 6))
plt.plot(grouped_data['Correct Rate'], grouped_data['Total Expanded Num'], marker='o')
plt.xlabel('Correct Rate')
plt.ylabel('Average Expanded Num')
plt.title('Average Expanded Num vs Correct Rate')
plt.grid(True)
plt.show()

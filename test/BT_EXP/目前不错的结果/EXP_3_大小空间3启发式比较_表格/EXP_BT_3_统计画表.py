import pandas as pd

# 假设 results_df 是之前代码生成的包含所有结果的 DataFrame
# 载入DataFrame
filename = "EXP_BT_3_big_small_3algo_table_001_record_timeout"
results_df = pd.read_csv(f"{filename}.csv")

# 定义mode和heuristic的映射
# Define mappings for mode and heuristic
mode_mapping = {
    'user-defined': 'Large Custom Space',
    'small-predicate-objs': 'Small Space'
}
heuristic_mapping = {
    -1: '-1 No Heuristic',
    0: '0 Non-optimal Fast Heuristic',
    1: '1 Optimal Heuristic'
}

# 映射mode和heuristic
results_df['mode'] = results_df['Mode'].map(mode_mapping)
results_df['heuristic'] = results_df['Heuristic Choice'].map(heuristic_mapping)

# 使用一个字典来指定哪些度量基于全部数据计算，哪些基于 Time Limit Exceeded 为 0
metrics_mode = {
    'Expanded Num': 'no_timeout',
    'Planning Time Total': 'no_timeout',
    'Current Cost': 'no_timeout',
    'Error': 'all',
    'Time Limit Exceeded': 'all'
}

# 根据 metrics_mode 过滤数据
filtered_df = results_df[results_df['Time Limit Exceeded'] == 0]
all_data_df = results_df

# 初始化一个空的DataFrame来存储最终结果
final_results = pd.DataFrame()

# 循环处理每个度量，根据 metrics_mode 决定使用的数据集
for metric, mode in metrics_mode.items():
    if mode == 'no_timeout':
        # 使用过滤后的数据集计算平均值
        temp_results = filtered_df.groupby(['mode', 'heuristic']).agg({metric: 'mean'}).rename(columns={metric: metric})
    else:
        # 使用全部数据集计算平均值
        temp_results = all_data_df.groupby(['mode', 'heuristic']).agg({metric: 'mean'}).rename(columns={metric: metric})

    # 如果 final_results 是空的，直接赋值
    if final_results.empty:
        final_results = temp_results
    else:
        # 否则，合并当前结果
        final_results = final_results.join(temp_results)

# 重新设置索引，方便查看
final_results = final_results.reset_index()

# 输出到控制台
# 假设 avg_results 是你要打印的 DataFrame
# 设置 Pandas 显示选项以确保打印整个 DataFrame
pd.set_option('display.max_rows', None)       # 设置为 None 以显示所有行
pd.set_option('display.max_columns', None)   # 设置为 None 以显示所有列
pd.set_option('display.width', None)         # 自动检测控制台的宽度
pd.set_option('display.max_colwidth', None)  # 显示每列的完整内容
print(final_results)
print("----------------------")
formatted_string = final_results.to_csv(sep='\t', index=False)
print(formatted_string)

# 保存到CSV
final_results.to_csv(f"{filename}_Avg_summary.csv", index=False)
print(f"Data has been saved to '{filename}_Avg_summary.csv'")
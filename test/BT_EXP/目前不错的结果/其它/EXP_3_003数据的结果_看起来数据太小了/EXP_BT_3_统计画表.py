import pandas as pd

# 假设 results_df 是之前代码生成的包含所有结果的 DataFrame
# 载入DataFrame
filename = "EXP_BT_3_big_small_3algo_table"
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

# 计算每个组合的平均值
avg_results = results_df.groupby(['mode', 'heuristic']).agg({
    'Expanded Num': 'mean',
    'Planning Time Total': 'mean',
    'Current Cost': 'mean',
    'Error': 'mean',
    'Time Limit Exceeded': 'mean'
}).reset_index()

# 重命名列以匹配需求
avg_results.rename(columns={
    'Expanded Num': 'expanded_num',
    'Planning Time Total': 'time',
    'Current Cost': 'cost',
    'Error': 'error',
    'Time Limit Exceeded': 'timeout'
}, inplace=True)

# 输出到控制台
# 假设 avg_results 是你要打印的 DataFrame
# 设置 Pandas 显示选项以确保打印整个 DataFrame
pd.set_option('display.max_rows', None)       # 设置为 None 以显示所有行
pd.set_option('display.max_columns', None)   # 设置为 None 以显示所有列
pd.set_option('display.width', None)         # 自动检测控制台的宽度
pd.set_option('display.max_colwidth', None)  # 显示每列的完整内容
# 打印 DataFrame
print(avg_results)
print("----------------------")
formatted_string = avg_results.to_csv(sep='\t', index=False)
# 打印出来以便复制
print(formatted_string)

# 保存到CSV
avg_results.to_csv(f"{filename}_Avg_summary.csv", index=False)
print(f"Data has been saved to '{filename}_Avg_summary.csv'")
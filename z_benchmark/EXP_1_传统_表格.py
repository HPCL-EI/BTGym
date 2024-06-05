import copy
import os
import random
from btgym.utils import ROOT_PATH
import pandas as pd

algorithms = ['opt_h0', 'opt_h0_llm','obtea', 'bfs','weak']  # , 'opt_h1','opt_h1_llm'
scenes = ['RH', 'VH', 'RHS', 'RW']  # 'RH', 'VH', 'RHS', 'RW'

# 创建一个空的 DataFrame 用来收集数据
# 初始化空的 DataFrame
results_df = pd.DataFrame(columns=[
    'Difficulty', 'Scene', 'Algorithm', 'Success Rate', 'Timeout Rate',
    'Average Expanded Number', 'Average Planning Time', 'Average Current Cost',
    'Average Action Number', 'Average Tick Time'
])

for difficulty in ["single","multi"]:
    for scene in scenes:
        for alg_str in algorithms:
            data_path = f"{ROOT_PATH}/../z_benchmark/algo_details/{difficulty}_{scene}_{alg_str}.csv"
            data = pd.read_csv(data_path)

            # 统计成功率
            success_rate = (data['Error'] == False).mean()

            # 统计超时率 Time_Limit_Exceeded=True的占比
            timeout_rate = (data['Time_Limit_Exceeded'] == True).mean()

            # 计算平均  Expanded_Number、Planning_Time_Total
            average_expanded_number = round(data['Expanded_Number'].mean(),3)
            average_planning_time_total = round(data['Planning_Time_Total'].mean(),3)

            # 计算平均 Current_Cost、Action_Number、Tick_Time 如果不成功，这三者设置为最大值
            # data.loc[data['Error'] == True,\
            #     ['Current_Cost', 'Action_Number', 'Tick_Time']] = data[['Current_Cost', 'Action_Number', 'Tick_Time']].max().max()
            # data.loc[data['Error'] == True, ['Current_Cost']] = data[['Current_Cost']].max().max()
            # data.loc[data['Error'] == True,['Action_Number']] = data[['Action_Number']].max().max()
            # data.loc[data['Error'] == True,['Tick_Time']] = data[['Tick_Time']].max().max()
            #
            # average_current_cost = data['Current_Cost'].mean()
            # average_action_number = data['Action_Number'].mean()
            # average_tick_time = data['Tick_Time'].mean()

            # 对于不成功的情况，将 Current_Cost、Action_Number、Tick_Time 设置为最大值
            data.loc[data['Error'] == True, 'Current_Cost'] = data['Current_Cost'].max()
            data.loc[data['Error'] == True, 'Action_Number'] = data['Action_Number'].max()
            data.loc[data['Error'] == True, 'Tick_Time'] = data['Tick_Time'].max()

            # 只对成功的案例计算平均值（即 Error 为 False 的情况）
            average_current_cost = round(data.loc[data['Error'] == False, 'Current_Cost'].mean(),2)
            average_action_number = round(data.loc[data['Error'] == False, 'Action_Number'].mean(),2)
            average_tick_time = round(data.loc[data['Error'] == False, 'Tick_Time'].mean(),2)


            # 输出结果
            # print("成功率:", success_rate)
            # print("超时率:", timeout_rate)
            # print("平均展开节点数:", average_expanded_number)
            # print("平均计划时间总和:", average_planning_time_total)
            # print("平均当前成本:", average_current_cost)
            # print("平均动作数:", average_action_number)
            # print("平均时钟时间:", average_tick_time)

            # 添加数据到 DataFrame
            # 构建新行
            new_row = pd.DataFrame([{
                'Difficulty': difficulty,
                'Scene': scene,
                'Algorithm': alg_str,
                'Success Rate': success_rate,
                'Timeout Rate': timeout_rate,
                'Average Expanded Number': average_expanded_number,
                'Average Planning Time': average_planning_time_total,
                'Average Current Cost': average_current_cost,
                'Average Action Number': average_action_number,
                'Average Tick Time': average_tick_time
            }])
            # 使用 concat 添加到 DataFrame
            results_df = pd.concat([results_df, new_row], ignore_index=True)

# 将结果保存到 CSV 文件
results_df.to_csv('./algorithm_performance_results.csv', index=False)



# 打印全部结果
# 设置 Pandas 显示选项，以便完整显示 DataFrame
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 自动行宽
pd.set_option('display.max_colwidth', None)  # 显示完整列内容
# 按 'Difficulty' 和 'Scene' 排序数据
results_df.sort_values(by=['Difficulty', 'Scene'], inplace=True)
# 循环遍历每个 'Difficulty' 分组
for difficulty, group_df in results_df.groupby('Difficulty'):
    print(f"Difficulty: {difficulty}")
    print("-" * 80)

    # 再循环每个 'Scene' 分组
    for scene, scene_group in group_df.groupby('Scene'):
        print(f"Scene: {scene}")
        print(scene_group[
                  ['Algorithm', 'Success Rate', 'Timeout Rate', 'Average Expanded Number', 'Average Planning Time',
                   'Average Current Cost', 'Average Action Number', 'Average Tick Time']])
        print("\n")  # 添加空行以增加可读性

    print("=" * 80)  # 在每个 'Difficulty' 分组后打印分隔线


# results_df 转为 latex
# 按 'Difficulty' 和 'Scene' 排序数据
results_df.sort_values(by=['Difficulty', 'Scene'], inplace=True)

# 输出为 LaTeX 格式
latex_string = ""
for difficulty, group_df in results_df.groupby('Difficulty'):
    latex_string += f"% Difficulty: {difficulty}\n"
    latex_string += "\\begin{table}[ht]\n\\centering\n"
    latex_string += f"\\caption{{Performance Metrics for {difficulty}}}\n"
    latex_string += "\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n"
    latex_string += "Algorithm & SR (\%) & Timeout Rate & Expanded Number & Planning Time (s) & Current Cost & Action Number & Tick Time \\\\\n\\midrule\n"
    for scene, scene_group in group_df.groupby('Scene'):
        latex_string += f"\\multicolumn{{8}}{{l}}{{\\textbf{{Scene: {scene}}}}} \\\\\n"  # Adding scene as a subheader
        latex_string += scene_group.to_latex(index=False, header=False, escape=False)
        latex_string += "\\midrule\n"  # Add midrule after each scene for separation
    latex_string += "\\bottomrule\n"
    latex_string += "\\end{tabular}\n"
    latex_string += "\\end{table}\n"
    latex_string += "\\clearpage\n"  # Optional: Force a page break after each table for clarity

# Print or save the LaTeX string
print(latex_string)

# Optionally, write the string to a .tex file
with open('table_output.tex', 'w') as f:
    f.write(latex_string)



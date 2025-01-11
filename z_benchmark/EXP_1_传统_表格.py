import copy
import os
import random
from btgym.utils import ROOT_PATH
import pandas as pd

algorithms = ['bfs','obtea','opt_h0_llm','opt_h0','hbtp']  # , 'opt_h1','opt_h1_llm'
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
results_df_sorted = results_df.sort_values(
    by=['Difficulty', 'Scene'],
    key=lambda col: col.map({'single': 0, 'multi': 1}) if col.name == 'Difficulty' else col.map({'RW': 0, 'VH': 1, 'RHS': 2, 'RH': 3})
)
print(results_df_sorted)
# LaTeX table template
latex_table = """
\\begin{{table}}[ht]
\\centering
\\caption{{Performance Metrics by Scene and Algorithm}}
\\label{{my-label}}
\\begin{{tabular}}{{@{{}}lrrrrrrr@{{}}}}
\\toprule
\\textbf{{Algorithm}} & \\textbf{{SR}} & \\textbf{{Timeout}} & \\textbf{{Expanded}} & \\textbf{{Planning}} & \\textbf{{Current}} & \\textbf{{Action}} & \\textbf{{Ticks}} \\\\
                   &             & \\textbf{{Rate}}    & \\textbf{{Number}} & \\textbf{{Time}}          & \\textbf{{Cost}}         & \\textbf{{Number}}      & \\textbf{{Time}}      \\\\ \\midrule
\\multicolumn{{8}}{{l}}{{\\textbf{{Single-goal}}}} \\\\
\\midrule
\\multicolumn{{8}}{{l}}{{\\textbf{{Scenario: RoboWaiter}}}} \\\\
{single_rw_rows}
\\midrule
\\multicolumn{{8}}{{l}}{{\\textbf{{Scenario: VirtualHome}}}} \\\\
{single_vh_rows}
\\midrule
\\multicolumn{{8}}{{l}}{{\\textbf{{Scenario: RobotHow-Small}}}} \\\\
{single_rhs_rows}
\\midrule
\\multicolumn{{8}}{{l}}{{\\textbf{{Scenario: RobotHow}}}} \\\\
{single_rh_rows}
\\midrule
\\multicolumn{{8}}{{l}}{{\\textbf{{Multi-goal}}}} \\\\
\\midrule
\\multicolumn{{8}}{{l}}{{\\textbf{{Scenario: RoboWaiter}}}} \\\\
{multi_rw_rows}
\\midrule
\\multicolumn{{8}}{{l}}{{\\textbf{{Scenario: VirtualHome}}}} \\\\
{multi_vh_rows}
\\midrule
\\multicolumn{{8}}{{l}}{{\\textbf{{Scenario: RobotHow-Small}}}} \\\\
{multi_rhs_rows}
\\midrule
\\multicolumn{{8}}{{l}}{{\\textbf{{Scenario: RobotHow}}}} \\\\
{multi_rh_rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
def escape_underscores(s):
    return s.replace('_', '\\_').replace('nan', '-')



def create_latex_rows(df, difficulty, scene):
    rows = []
    subset_df = df[(df['Difficulty'] == difficulty) & (df['Scene'] == scene)]
    for _, row in subset_df.iterrows():
        avg_current_cost = row['Average Current Cost'] if row['Average Current Cost'] is not None else '-'
        avg_action_number = row['Average Action Number'] if row['Average Action Number'] is not None else '-'
        avg_tick_time = row['Average Tick Time'] if row['Average Tick Time'] is not None else '-'
        rows.append(f"{row['Algorithm']} & {row['Success Rate']*100:.0f}\\% & {row['Timeout Rate']*100:.0f}\\% & {row['Average Expanded Number']:.2f} & {row['Average Planning Time']:.3f} & {avg_current_cost} & {avg_action_number} & {avg_tick_time} \\\\")
    return "\n".join(rows)

# scenes = ['RW', 'VH', 'RHS', 'RH']
# difficulties = ['single', 'multi']
#
# single_goal_rows = []
# multi_goal_rows = []
#
# for scene in scenes:
#     single_goal_rows.append(f"\\multicolumn{{8}}{{l}}{{\\textbf{{Scenario: {scene}}}}} \\\\")
#     single_goal_rows.append(create_latex_rows(results_df_sorted, 'single', scene))
#     multi_goal_rows.append(f"\\multicolumn{{8}}{{l}}{{\\textbf{{Scenario: {scene}}}}} \\\\")
#     multi_goal_rows.append(create_latex_rows(results_df_sorted, 'multi', scene))
#
# single_goal_rows_str = "\n\\midrule\n".join(single_goal_rows)
# multi_goal_rows_str = "\n\\midrule\n".join(multi_goal_rows)
#
# latex_table = latex_table.format(
#     single_goal_rows=single_goal_rows_str,
#     multi_goal_rows=multi_goal_rows_str
# )

single_rw_rows = create_latex_rows(results_df_sorted, 'single', 'RW')
single_vh_rows = create_latex_rows(results_df_sorted, 'single', 'VH')
single_rhs_rows = create_latex_rows(results_df_sorted, 'single', 'RHS')
single_rh_rows = create_latex_rows(results_df_sorted, 'single', 'RH')
multi_rw_rows = create_latex_rows(results_df_sorted, 'multi', 'RW')
multi_vh_rows = create_latex_rows(results_df_sorted, 'multi', 'VH')
multi_rhs_rows = create_latex_rows(results_df_sorted, 'multi', 'RHS')
multi_rh_rows = create_latex_rows(results_df_sorted, 'multi', 'RH')

latex_table = latex_table.format(
    single_rw_rows=single_rw_rows,
    single_vh_rows=single_vh_rows,
    single_rhs_rows=single_rhs_rows,
    single_rh_rows=single_rh_rows,
    multi_rw_rows=multi_rw_rows,
    multi_vh_rows=multi_vh_rows,
    multi_rhs_rows=multi_rhs_rows,
    multi_rh_rows=multi_rh_rows
)

latex_table = escape_underscores(latex_table)

print(latex_table)
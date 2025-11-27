import pandas as pd
import os
import sys

base_dir = 'btpg_20250506/z_experience_results/exp4_Attraction/'

settings = [
    {
        'label': r'$\epsilon_1=0$',
        'sublabel': r'$\epsilon_2=0.2$',
        'single': '2_changes_all_p=0.2_t=5_d=100_single.csv',
        'multi': '2_changes_all_p=0.2_t=5_d=100_multi.csv'
    },
    {
        'label': r'$\epsilon_1=0$',
        'sublabel': r'$\epsilon_2=0.5$',
        'single': '2_changes_all_p=0.5_t=5_d=100_single.csv',
        'multi': '2_changes_all_p=0.5_t=5_d=100_multi.csv'
    },
    {
        'label': r'$\epsilon_1=1$',
        'sublabel': r'$\epsilon_2=0$',
        'single': '1_only_changes_initial_t=5_False_single.csv',
        'multi': '1_only_changes_initial_t=5_False_multi.csv'
    },
    {
        'label': r'$\epsilon_1=1$',
        'sublabel': r'$\epsilon_2=0.2$',
        'single': '3_initial_changes_all_single_p=0.2_t=5.csv',
        'multi': '3_initial_changes_all_multi_p=0.2_t=5.csv'
    }
]

algo_map = {
    'BT Expansion': 'bfs_F',
    'OBTEA': 'obtea_F',
    'HBTP': 'opt_h0_F',
    'UHBTP': 'hbtp_F',
    'HBTP-Oracle': 'opt_h0_llm_F'
}

algos_order = ['BT Expansion', 'OBTEA', 'HBTP', 'UHBTP', 'HBTP-Oracle']

def get_values(filepath):
    if not os.path.exists(filepath):
        return {algo: ['-', '-', '-', '-'] for algo in algos_order}
        
    df = pd.read_csv(filepath, index_col=0)
    df.index = df.index.str.strip()
    data = {}
    for algo_name in algos_order:
        csv_key = algo_map[algo_name]
        if csv_key in df.index:
            row = df.loc[csv_key]
            try:
                rw = int(round(row['RW'] * 100))
            except: rw = '-'
            try:
                vh = int(round(row['VH'] * 100))
            except: vh = '-'
            try:
                # Handle RHS or OG column naming
                if 'RHS' in row:
                    og = int(round(row['RHS'] * 100))
                elif 'OG' in row:
                    og = int(round(row['OG'] * 100))
                else:
                    og = '-'
            except: og = '-'
            try:
                rh = int(round(row['RH'] * 100))
            except: rh = '-'
            
            data[algo_name] = [rw, vh, og, rh]
        else:
            data[algo_name] = ['-', '-', '-', '-']
    return data

print(r'\begin{table}[t]')
print(r'\centering')
print(r'\setlength{\tabcolsep}{2pt} % 调整列间距，默认为6pt')
print(r'')
print(r'\vspace{0.1cm} % 在表格上方添加1厘米的垂直空间')
print(r'')
print(r'\small')
print(r'\begin{tabular}{@{}llcccccccc@{}}')
print(r'\toprule')
print(r'% \multicolumn{1}{c}{Setting} & \multicolumn{1}{c}{Algorithm}')
print(r'\multirow{2}{*}{\textbf{Settings}} & \multirow{2}{*}{\textbf{Algorithms}} & \multicolumn{4}{c}{\textbf{Single-Goal}} & \multicolumn{4}{c}{\textbf{Multi-Goal}} \\ \cmidrule(lr){3-6} \cmidrule(l){7-10}')
print(r' & & \textbf{RW} & \textbf{VH} & \textbf{OG} & \textbf{RH} & \textbf{RW} & \textbf{VH} & \textbf{OG} & \textbf{RH} \\ \midrule')

for i, setting in enumerate(settings):
    s_path = os.path.join(base_dir, setting['single'])
    m_path = os.path.join(base_dir, setting['multi'])
    
    s_data = get_values(s_path)
    m_data = get_values(m_path)
    
    print(r'         ')
    
    # Row 1: Empty label
    algo = algos_order[0]
    s_vals = s_data[algo]
    m_vals = m_data[algo]
    print(f"         & {algo} & {s_vals[0]} & {s_vals[1]} & {s_vals[2]} & {s_vals[3]} & {m_vals[0]} & {m_vals[1]} & {m_vals[2]} & {m_vals[3]} \\\\")
    
    # Row 2: label
    algo = algos_order[1]
    s_vals = s_data[algo]
    m_vals = m_data[algo]
    print(f"{setting['label']}  & {algo} & {s_vals[0]} & {s_vals[1]} & {s_vals[2]} & {s_vals[3]} & {m_vals[0]} & {m_vals[1]} & {m_vals[2]} & {m_vals[3]} \\\\")
    
    # Row 3: sublabel
    algo = algos_order[2]
    s_vals = s_data[algo]
    m_vals = m_data[algo]
    print(f"{setting['sublabel']} & {algo} & {s_vals[0]} & {s_vals[1]} & {s_vals[2]} & {s_vals[3]} & {m_vals[0]} & {m_vals[1]} & {m_vals[2]} & {m_vals[3]} \\\\")
    
    # Row 4+: Empty
    for algo in algos_order[3:]:
        s_vals = s_data[algo]
        m_vals = m_data[algo]
        print(f"         & {algo} & {s_vals[0]} & {s_vals[1]} & {s_vals[2]} & {s_vals[3]} & {m_vals[0]} & {m_vals[1]} & {m_vals[2]} & {m_vals[3]} \\\\")
    
    if i < len(settings) - 1:
        print(r'\midrule')

print(r'\bottomrule')
print(r'\end{tabular}')
print(r'% \normalsize % 恢复到正常字体大小')
print(r'\caption{The success rates (\%) of planned BT in noisy environments, which reflect the Execution Robustness metric.}')
print(r'\label{tab:performance_metrics}')
print(r'\end{table}')


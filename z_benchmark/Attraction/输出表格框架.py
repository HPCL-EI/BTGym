import pandas as pd
import random


def get_SR(scene, algo_str, just_best):
    return round(random.random(), 2)

algorithms = ['opt_h0', 'opt_h1', 'obtea', 'bfs', 'dfs']  # 'opt_h0', 'opt_h1', 'obtea', 'bfs', 'dfs'
scenes = ['RH', 'RHS', 'RW', 'VH']  # 'RH', 'RHS', 'RW', 'VH'
just_best_bts = [True, False] # True, False

# 创建df
index = [f'{algo_str}_{tb}' for tb in ['T', 'F'] for algo_str in algorithms ]
df = pd.DataFrame(index=index, columns=scenes)
for just_best in just_best_bts:
    for algo_str in algorithms:
        index_key = f'{algo_str}_{"T" if just_best else "F"}'
        for scene in scenes:
            df.at[index_key, scene] = get_SR(scene, algo_str, just_best)

formatted_string = df.to_csv(sep='\t')
print(formatted_string)
print("----------------------")
print(df)




import re
import os
import faiss
import numpy as np
from btgym.utils import ROOT_PATH
from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
from btgym.algos.llm_client.vector_database_env_goal import check_index_exists,parse_and_prepare_data,embed_and_store
def list_all_goals(index_path):
    """输出数据库中所有的 goals"""
    # 读取存储的元数据
    metadata = np.load(index_path.replace(".index", "_metadata.npy"), allow_pickle=True)

    # 提取并输出所有的 goals
    all_goals = [entry['value']['Goals'] for entry in metadata]
    return all_goals


if __name__ == '__main__':
    llm = LLMGPT3()

    filename = "Group0"
    index_path = f"{ROOT_PATH}/../test/dataset/DATABASE/{filename}_env_goal_vectors.index"

    # 输出数据库中的所有 goals
    goals = list_all_goals(index_path)
    for goal in goals:
        print(goal)

    print(f"共有 {len(goals)} 数据")

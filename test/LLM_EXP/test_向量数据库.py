import faiss
import numpy as np
from btgym.utils import ROOT_PATH
from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
import re
import os

def parse_and_prepare_data(file_path):
    """从文本文件中解析数据，生成键值对"""
    data = {}
    current_id = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.isdigit():
                current_id = line
                data[current_id] = {"Environment": "", "Instruction": "", "Goals": "", "Actions": "", "Key_Predicate": "", "Key_Objects": ""}
            else:
                match = re.match(r"(\w+):\s*(.*)", line)
                if match and current_id:
                    key, value = match.groups()
                    data[current_id][key] = value

    # 将 Environment 和 Instruction 组合作为键
    keys = [f"{entry['Environment']}: {entry['Instruction']}" for entry in data.values()]
    return keys, data

def extract_embedding_vector(response):
    """从 CreateEmbeddingResponse 对象中提取嵌入向量"""
    if response and len(response.data) > 0:
        # 提取第一个嵌入对象的向量数据
        return response.data[0].embedding
    else:
        raise ValueError("Empty or invalid embedding response.")

def embed_and_store(llm, keys, data, index_path):
    """使用 llm.embedding 方法生成嵌入并存储在向量数据库中"""

    # 假设 keys 是一组文本键值
    embeddings = np.array([extract_embedding_vector(llm.embedding(key)) for key in keys], dtype='float32')
    # 将嵌入和索引保存到 Faiss 索引
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)

    # 保存其他相关数据
    # np.save(index_path.replace(".index", "_metadata.npy"), list(data.keys()))
    # 保存其他相关数据，包括每个 key 及其对应的 value
    metadata = [{"key": key, "value": data[key]} for key in data.keys()]
    np.save(index_path.replace(".index", "_metadata.npy"), metadata)


def search_similar(index_path, llm, environment, instruction, top_n=3):
    """搜索与给定环境和指令组合最相似的记录"""
    index = faiss.read_index(index_path)
    metadata = np.load(index_path.replace(".index", "_metadata.npy"), allow_pickle=True)

    query = f"{environment}: {instruction}"
    query_embedding = np.array([extract_embedding_vector(llm.embedding(query))], dtype='float32')
    distances, indices = index.search(query_embedding, top_n)

    # results = [{"id": metadata[idx], "distance": dist} for dist, idx in zip(distances[0], indices[0])]
    results = [{"id": idx, "distance": dist, "key": metadata[idx]['key'], "value": metadata[idx]['value']}
               for dist, idx in zip(distances[0], indices[0])]
    return results

def check_index_exists(index_path):
    """检查索引文件和元数据文件是否存在"""
    index_file = index_path
    metadata_file = index_path.replace(".index", "_metadata.npy")
    return os.path.exists(index_file) and os.path.exists(metadata_file)


# 示例数据路径和索引文件路径
file_path = f"{ROOT_PATH}/../test/dataset/database_cys_5.txt"
index_path = 'env_instruction_vectors.index'
should_rebuild_index = False  # 是否重新生成数据库的标志


# 假设 llm 是已经初始化的嵌入模型对象
# 你可以根据实际的模型对象进行替换和设置
llm = LLMGPT3()
# 检查文件存在或决定是否重建
if should_rebuild_index or not check_index_exists(index_path):
    keys, data = parse_and_prepare_data(file_path)
    embed_and_store(llm, keys, data, index_path)


# 使用问题进行检索
environment = "4"
# instruction = "Put the bag of chips on the corner of my desk."
instruction = "Wash the bananas, cut the bananas and put it in the fridge"
results = search_similar(index_path, llm, environment, instruction,top_n=5)

# 输出检索结果
for result in results:
    record_id = result['id']
    distance = result['distance']
    key = result['key']
    value = result['value']
    print(f"Record ID: {record_id}, Distance: {distance}")
    print(f"Key: {key}, Value: {value}\n")

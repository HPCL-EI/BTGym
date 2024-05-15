def read_env_file(file_path):
    env_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # 跳过空行
                key, value = line.split(maxsplit=1)
                env_dict[key] = set(value.strip().split(', '))
    return env_dict

# 使用提供的文件路径读取并解析环境文件
file_path = '/mnt/data/ENV_5.txt'
env_dict = read_env_file(file_path)

# 示例：检索 key="1" 的环境设置
key = "1"
if key in env_dict:
    print(f"Environment settings for key {key}: {env_dict[key]}")
else:
    print(f"Key {key} not found in the environment dictionary.")

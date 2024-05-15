import re


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    return data


def save_data(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(data)


def extract_goals(data):
    goals = re.findall(r'Goals: (.+)', data)
    return set(goals)


def filter_test_data(test_data, train_goals):
    test_entries = re.split(r'\n\n(?=\d+\nEnvironment)', test_data)
    filtered_entries = [entry for entry in test_entries if re.search(r'Goals: (.+)', entry).group(1) not in train_goals]
    return filtered_entries

def main():
    # 文件路径
    train_file_path = 'train_data_400.txt'
    test_file_path = 'test_data_40.txt'
    output_test_file_path = 'filtered_test_data.txt'

    # 加载数据
    train_data = load_data(train_file_path)
    test_data = load_data(test_file_path)

    # 提取训练集和测试集中的Goals
    train_goals = extract_goals(train_data)
    test_goals = extract_goals(test_data)

    # 打印提取的Goals数量
    print(f"Number of goals in training data: {len(train_goals)}")
    print(f"Number of goals in test data: {len(test_goals)}")

    # 去重
    filtered_test_entries = filter_test_data(test_data, train_goals)

    # 打印最终测试数据的条数
    print(f"Number of entries in the final test data: {len(filtered_test_entries)}")

    # 将过滤后的数据重新组合为字符串
    filtered_test_data = '\n\n'.join(filtered_test_entries)

    # 保存新的测试集
    save_data(filtered_test_data, output_test_file_path)
    print(f"Filtered test data saved to {output_test_file_path}")


if __name__ == "__main__":
    main()
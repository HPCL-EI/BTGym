"""
example =
[
    {
        'Instruction': 'Place the chicken on the kitchentable.',
        'Goals': 'IsOn_chicken_kitchentable',
        'Actions': ['Walk_chicken', 'RightGrab_chicken', 'Walk_kitchentable', 'RightPut_chicken_kitchentable']
    },
    {
        'Instruction': 'Grab the book from the desk and put it on the nightstand.',
        'Goals': 'IsOn_book_nightstand',
        'Actions': ['Walk_desk', 'RightGrab_book', 'Walk_nightstand', 'RightPut_book_nightstand']
    },
]
"""
def read_dataset(filename='./dataset.txt'):
    with open(filename, 'r') as file:
        # 读取文件内容
        lines = file.readlines()
    # print(lines)
    dataset = []

    # 遍历所有行，每5行分一个块
    for i in range(0, len(lines), 5):
        five_lines = lines[i:i + 5]
        five_lines = [line.strip() for line in five_lines]
        five_lines = five_lines[1:]
        dataset.append(five_lines)

    example = []
    # print(dataset)
    # 打印分隔后的内容
    for index, item in enumerate(dataset):
        example_number = f"example {index + 1}:"
        # print(example_number)
        item = '\n'.join(item)
        # print(item)
        dict = {}
        parts = item.strip().split('\n')
        # 解析每个部分并添加到字典中
        for part in parts:
            # print(example_number,'part', part)
            key, value = part.split(':', 1)
            key = key.strip()  # 移除键名两侧的空格
            value = value.strip()  # 移除值两侧的空格
            dict[key] = value

        # 如果Actions的值包含多个动作，则分割成列表
        if 'Actions' in dict:
            dict['Actions'] = dict['Actions'].split(', ')

        # 打印字典
        # print(dict)
        example.append(dict)
        # print()  # 打印一个空行作为块之间的分隔
    return example

if __name__ == '__main__':
    example = read_dataset('./dataset.txt')
    print(example[0])
    print(example[1])

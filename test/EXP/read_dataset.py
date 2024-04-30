"""
读取返回一个列表：
example =
[
    {
        'Environment': 2,
        'Instruction': 'Place the apple on the kitchen counter and make sure the kitchen cabinet is open.',
        'Goals': ['IsOn_apple_kitchencounter', 'IsOpen_kitchencabinet'],
        'Actions': ['Walk_apple', 'RightGrab_apple', 'Walk_kitchencounter', 'RightPut_apple_kitchencounter'],
        'Key_Predicate': ['Walk', 'RightGrab', 'RightPut'],
        'Key_Object': ['apple', 'kitchencounter']
    }
    ......
]
"""


def read_dataset(filename='./dataset.txt'):
    with open(filename, 'r') as f:
        # 读取文件内容
        lines = f.readlines()
    dataset = []

    # 遍历所有行，每8行分一个块
    for i in range(0, len(lines), 8):
        _8_lines = lines[i:i + 8]
        _8_lines = [line.strip() for line in _8_lines]
        _8_lines = _8_lines[1:]
        dataset.append(_8_lines)

    example = []
    for index, item in enumerate(dataset):
        item = '\n'.join(item)
        dict = {}
        parts = item.strip().split('\n')
        for part in parts:
            key, value = part.split(':', 1)
            key = key.strip()
            value = value.strip()
            dict[key] = value
            if key == 'Environment':
                v = int(value)
                dict[key] = v
            if key in ('Key_Predicate', 'Key_Object', 'Actions'):
                parts = [v.strip() for v in value.split(",")]
                dict[key] = parts
            if key == 'Goals':
                parts = [v.strip() for v in value.split("&")]
                dict[key] = parts
        example.append(dict)
    return example


if __name__ == '__main__':
    example = read_dataset('C:/Users/yangz/Desktop/dataset0429.txt')
    print(example[0])


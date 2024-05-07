
def count_accuracy(expected, actual):
    correct = 0
    incorrect = 0

    # 统计正确个数和错误个数
    for item in actual:
        if item in expected:
            correct += 1
        else:
            incorrect += 1

    # 计算正确率和错误率
    total = correct + incorrect
    accuracy = (correct / len(expected)) * 100
    error_rate = (incorrect / len(expected)) * 100

    return correct, incorrect, accuracy, error_rate
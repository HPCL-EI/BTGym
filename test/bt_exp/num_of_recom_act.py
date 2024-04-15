import numpy as np

from btgym.utils import ROOT_PATH
from dataset import read_dataset
'''
Place the chicken on the kitchentable.
IsOn_chicken_kitchentable
['Walk_chicken', 'RightGrab_chicken', 'Walk_kitchentable', 'RightPut_chicken_kitchentable']
'''

# Import dataset
dataset = read_dataset(f"{ROOT_PATH}\\..\\test\\bt_exp\\dataset.txt")
# print(dataset)

ratio_ls = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
result = np.zeros((len(ratio_ls), len(ratio_ls)))

# correct_ratio, error_ratio

for index, example in enumerate(dataset):
    if index > 1:
        break
    Instruction = example['Instruction']
    Goals = example['Goals']
    Actions = example['Actions']

    # print(Instruction)
    # print(Goals)
    # print(Actions)

    priority_act_ls = []






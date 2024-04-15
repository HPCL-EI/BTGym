import pickle

# 替换'file_path.pkl'为你的.pkl文件路径
file_path = 'graph_input.pkl'

# 打开文件并加载内容
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# 确保加载的数据是字典类型
if isinstance(data, dict):
    print("成功读取字典:")
    # print(data)
else:
    print("文件内容不是字典")

data = data['nodes']

prop_obj_dic={'SURFACES':[],'SITTABLE':[], # put、sit
              'CAN_OPEN':[],'CONTAINERS':[], # close、putin
              'GRABBABLE':[],'HAS_SWITCH':[],  # grab、switch

              }

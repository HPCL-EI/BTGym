
from btgym.algos.bt_autogen.Action import Action
from btgym.utils.read_dataset import read_dataset
from btgym.utils import ROOT_PATH
from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction
import pickle

def collect_action_nodes(behavior_lib):
    action_list = []

    for cls in behavior_lib["Action"].values():
        if cls.can_be_expanded:
            print(f"可扩展动作：{cls.__name__}, 存在{len(cls.valid_args)}个有效论域组合")
            if cls.num_args == 0:
                action_list.append(Action(name=cls.get_ins_name(), **cls.get_info()))
            if cls.num_args == 1:
                for arg in cls.valid_args:
                    action_list.append(Action(name=cls.get_ins_name(arg), **cls.get_info(arg)))
            if cls.num_args > 1:
                for args in cls.valid_args:
                    action_list.append(Action(name=cls.get_ins_name(*args), **cls.get_info(*args)))

    print(f"共收集到{len(action_list)}个实例化动作:")
    # for a in self.action_list:
    #     if "Turn" in a.name:
    #         print(a.name)
    print("--------------------\n")

    return action_list


def refresh_VHT_samll_data():
    # 读入数据集合
    data_path = f"{ROOT_PATH}/../test/dataset/data0429.txt"
    data = read_dataset(data_path)
    data_num = len(data)
    print(f"导入 {data_num} 条数据")
    print(data[0])

    # 数据集中涉及的所有物体集合
    objs=set()
    for d in data:
        objs |= set(d['Key_Object'])

    categories = ['SURFACES', 'SITTABLE', 'CAN_OPEN', 'CONTAINERS', 'GRABBABLE', 'cleaning_tools', \
             'cutting_tools', 'HAS_SWITCH', 'HAS_PLUG', 'CUTABLE', 'EATABLE', 'WASHABLE', 'RECIPIENT', \
             'POURABLE', 'DRINKABLE']
    categories_objs_dic={}
    for ctg in categories:
        categories_objs_dic[ctg] = getattr(VHTAction, ctg)
        categories_objs_dic[ctg] &= objs


    ctg_objs_path = f"{ROOT_PATH}/../test/EXP/ctg_objs.pickle"
    # 打开一个文件用于写入，注意'b'表示二进制模式
    with open(ctg_objs_path, 'wb') as file:
        # 使用pickle.dump()函数将数据写入文件
        pickle.dump(categories_objs_dic, file)
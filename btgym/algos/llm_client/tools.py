

from sympy import symbols, Not, Or, And, to_dnf
from sympy import symbols, simplify_logic
import re


def act_str_process(act_str,already_split=False):

    if already_split:
        act_str_ls = act_str
    else:
        act_str_ls = act_str.replace(" ", "").split(",")

    priority_act_ls = []
    # rl_dic = {}
    # rightHandFull = False
    # leftHandFull = False
    #
    # need_to_again_ls=[]

    for literal in act_str_ls:
        literal = re.sub(r"[ ,()\[\] ]\n\\n", "", literal)

        if '_' in literal:
            first_part, rest = literal.split('_', 1)
            literal = first_part + '(' + rest
            # 添加 ')' 到末尾
            literal += ')'
            # 替换剩余的 '_' 为 ','
            literal = literal.replace('_', ',')
        priority_act_ls.append(literal)
    return priority_act_ls

        # if 'Grab' in literal:
        #     # 使用正则表达式匹配并提取括号内的内容 "Grab(milk)"
        #     matched = re.search(r"\((.*?)\)", literal)
        #     obj = matched.group(1) if matched else None
        #     if rightHandFull == False:
        #         rl_dic[obj] = "right"
        #         rightHandFull=True
        #         literal = "Right"+literal
        #     else:
        #         rl_dic[obj] = "left"
        #         leftHandFull=True
        #         literal = "Left" + literal

    #     need_to_again = False
    #     if "Put" in literal or "PutIn" in literal:
    #         matched = re.search(r"\((.*?)\)", literal)
    #         obj = matched.group(1) if matched else None
    #         if obj not in rl_dic.keys():
    #             need_to_again=True
    #             need_to_again_ls.append(literal)
    #         else:
    #             if rl_dic[obj]=="right":
    #                 rightHandFull = False
    #                 literal = "Right" + literal
    #             else:
    #                 leftHandFull = False
    #                 literal = "Left" + literal
    #     priority_act_ls.append(literal)
    #
    #
    # for literal in need_to_again_ls:
    #     matched = re.search(r"\((.*?)\)", literal)
    #     obj = matched.group(1) if matched else None
    #     if rl_dic[obj]=="right":
    #         rightHandFull = False
    #         literal = "Right" + literal
    #     else:
    #         leftHandFull = False
    #         literal = "Left" + literal
    #     priority_act_ls.append(literal)






def goal_transfer_str(goal):
    goal_dnf = str(to_dnf(goal, simplify=True))
    # print(goal_dnf)
    goal_set = []
    if ('|' in goal or '&' in goal or 'Not' in goal) or not '(' in goal:
        goal_ls = goal_dnf.split("|")
        for g in goal_ls:
            g_set = set()
            g = g.replace(" ", "").replace("(", "").replace(")", "")
            g = g.split("&")
            for literal in g:
                if '_' in literal:
                    first_part, rest = literal.split('_', 1)
                    literal = first_part + '(' + rest
                    # 添加 ')' 到末尾
                    literal += ')'
                    # 替换剩余的 '_' 为 ','
                    literal = literal.replace('_', ',')
                g_set.add(literal)
            goal_set.append(g_set)

    else:
        g_set = set()
        w = goal.split(")")
        g_set.add(w[0] + ")")
        if len(w) > 1:
            for x in w[1:]:
                if x != "":
                    g_set.add(x[1:] + ")")
        goal_set.append(g_set)
    return goal_set



def act_format_records(act_record_list):
    # 初始化一个空列表来存储格式化后的结果
    formatted_records = []
    predicate = []
    objects_ls= []
    # 遍历列表中的每个记录
    for record in act_record_list:

        if "," not in record:
            # 找到括号的位置
            start = record.find('(')
            end = record.find(')')
            # 提取动作和对象
            action = record[:start]
            obj = record[start+1:end]
            # 格式化为新的字符串格式
            formatted_record = f"{action}_{obj}"
            # 将格式化后的字符串添加到结果列表中
            formatted_records.append(formatted_record)
            predicate.append(action)
            objects_ls.append(obj)
        else:
            # 有逗号，即涉及两个物体
            start = record.find('(')
            end = record.find(')')
            action = record[:start]
            objects = record[start + 1:end].split(',')
            obj1 = objects[0].strip()  # 去除可能的空白字符
            obj2 = objects[1].strip()
            formatted_record = f"{action}_{obj1}_{obj2}"
            # 将格式化后的字符串添加到结果列表中
            formatted_records.append(formatted_record)
            predicate.append(action)
            objects_ls.append(obj1)
            objects_ls.append(obj2)
    return list(set(formatted_records)),list(set(predicate)),list(set(objects_ls))



def remove_duplicates_using_set(lst):
    return list(set(lst))


def update_objects_from_expressions(expressions, pattern, objects):
    for expr in expressions:
        match = pattern.search(expr)
        if match:
            # 将括号内的内容按逗号分割并加入到集合中
            objects.update(match.group(1).split(','))

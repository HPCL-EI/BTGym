

from sympy import symbols, Not, Or, And, to_dnf
from sympy import symbols, simplify_logic
import re


def act_str_process(act_str):

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



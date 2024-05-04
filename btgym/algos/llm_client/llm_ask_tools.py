import re
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process


def parse_llm_output(answer):


    goal_str = answer.split("Actions:")[0].replace("Goals:", "").strip()
    act_str = answer.split("Actions:")[1].split("Key Predicates:")[0].strip()
    predicate_str = answer.split("Key Predicates:")[1].split("Key Objects:")[0].strip()
    objects_str = answer.split("Key Objects:")[1].strip()

    goal_set = goal_transfer_str(goal_str)
    priority_act_ls = act_str_process(act_str)

    # Remove all spaces, Split by comma to create a list
    key_predicate = predicate_str.replace(" ", "").split(",")
    key_objects = objects_str.replace(" ", "").split(",")

    return goal_set,priority_act_ls,key_predicate,key_objects

def extract_initial_llm_outputs(llm,default_prompt_file,instuction,cur_cond_set):

    with open(default_prompt_file, 'r', encoding="utf-8") as f:
        prompt = f.read().strip()

    # 补充：向量数据库检索，拼接上最相近的 Example cur_cond_set
    # cur_env_state = ', '.join(map(str, cur_cond_set))
    # cur_data = instuction + "\n[current environmental condition]\n" + cur_env_state  # 可能还要再调整
    # cur_emb = llm.embedding(question=cur_data)
    # 导入向量数据库，找到最近的前5条。
    # 准备好的 30条数据 作为 向量数据库
    # example = ""
    # 将例子拼在后面
    # question+=example

    question = prompt+instuction

    messages = []
    messages.append({"role": "user", "content": question})
    answer = llm.request(message=messages)
    messages.append({"role": "assistant", "content": answer})
    print(answer)

    goal_set, priority_act_ls, key_predicate, key_objects = parse_llm_output(answer)

    print("goal",goal_set)
    print("act:",priority_act_ls)
    print("key_predicate",goal_set)
    print("key_objects:",priority_act_ls)


    # 提取目标中的所有物体
    objects = set()
    # 正则表达式用于找到括号中的内容
    pattern = re.compile(r'\((.*?)\)')
    # 遍历所有表达式，提取物体名称
    for expr in goal_set[0]:
        # 找到括号内的内容
        match = pattern.search(expr)
        if match:
            # 将括号内的内容按逗号分割并加入到集合中
            objects.update(match.group(1).split(','))
    key_objects += list(objects)
    key_objects = list(set(key_objects))

    return goal_set, priority_act_ls, key_predicate, key_objects, messages

def llm_reflect(llm, messages, reflect_prompt):
    messages.append({"role": "user", "content": reflect_prompt})
    answer = llm.request(message=messages)
    messages.append({"role": "assistant", "content": answer})

    print(answer)

    goal_set, priority_act_ls, key_predicate, key_objects = parse_llm_output(answer)

    print("goal",goal_set)
    print("act:",priority_act_ls)
    print("key_predicate",goal_set)
    print("key_objects:",priority_act_ls)

    return goal_set, priority_act_ls, key_predicate, key_objects, messages
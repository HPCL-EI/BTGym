import re
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process
from btgym.utils import ROOT_PATH
# 导入向量数据库检索的相关函数
from btgym.algos.llm_client.vector_database import search_nearest_examples



def parse_llm_output(answer,goals=True):

    if goals:
        goal_str = answer.split("Actions:")[0].replace("Goals:", "").strip()
        goal_set = goal_transfer_str(goal_str)
    else:
        goal_set=set()

    act_str = answer.split("Actions:")[1].split("Key_Predicates:")[0].strip()
    predicate_str = answer.split("Key_Predicates:")[1].split("Key_Objects:")[0].strip()
    objects_str = answer.split("Key_Objects:")[1].strip()
    priority_act_ls = act_str_process(act_str)

    # Remove all spaces, Split by comma to create a list
    key_predicate = predicate_str.replace(" ", "").split(",")
    key_objects = objects_str.replace(" ", "").split(",")

    if goals:
        return goal_set,priority_act_ls,key_predicate,key_objects
    else:
        return priority_act_ls, key_predicate, key_objects



def format_example(metadata):
    """格式化向量数据库的示例数据为所需的格式"""
    example_value = metadata['value']
    return (f"Instruction: {example_value['Instruction']}\n"
            f"Goals: {example_value['Goals']}\n"
            f"Actions: {example_value['Actions']}\n"
            f"Key Predicates: {example_value.get('Key_Predicates', '')}\n"
            f"Key Objects: {example_value['Key_Objects']}\n")

def extract_llm_from_instr_goal(llm,default_prompt_file,instruction,goals,cur_cond_set=None,\
                                choose_database=False,\
                                index_path=f"{ROOT_PATH}/../test/dataset/env_instruction_vectors.index",verbose=False):
    with open(default_prompt_file, 'r', encoding="utf-8") as f:
        prompt = f.read().strip()

    # 构建完整的 prompt，包括检索的 Examples 和当前的指令
    question = f"{prompt}\nInstruction: {instruction}\nGoals: {goals}"
    if verbose:
        print("============ Question ================\n",question)
    messages = []
    messages.append({"role": "user", "content": question})
    answer = llm.request(message=messages)
    messages.append({"role": "assistant", "content": answer})
    if verbose:
        print("============ Answer ================\n",answer)
    priority_act_ls, key_predicates, key_objects = parse_llm_output(answer,goals=False)

    return priority_act_ls, key_predicates, key_objects, messages


def extract_llm_from_instr(llm,default_prompt_file,instruction,cur_cond_set,\
                                choose_database=False,\
                                index_path=f"{ROOT_PATH}/../test/dataset/env_instruction_vectors.index"):
    """从向量数据库检索并生成初始 prompt"""

    with open(default_prompt_file, 'r', encoding="utf-8") as f:
        prompt = f.read().strip()

    if choose_database:
        # 补充：向量数据库检索，拼接上最相近的 Example cur_cond_set
        # cur_env_state = ', '.join(map(str, cur_cond_set))
        # cur_data = instuction + "\n[current environmental condition]\n" + cur_env_state  # 可能还要再调整
        # cur_emb = llm.embedding(question=cur_data)
        # 导入向量数据库，找到最近的前5条。
        # 准备好的 30条数据 作为 向量数据库
        # example = ""
        # 将例子拼在后面
        # question+=example
        # 检索向量数据库以获取最近的 Examples
        nearest_examples,distances = search_nearest_examples(index_path, llm, instruction,top_n=3)
        # 使用自定义的格式函数将检索到的示例格式化为目标样式
        example_texts = '\n'.join([format_example(ex) for ex in nearest_examples])
        example_texts = "[Examples]\n" + example_texts
        print("distances:",distances)
        # print("example_texts:\n",example_texts)
        # 替换 prompt 中的 [Examples] 部分
        example_marker = "[Examples]"
        if example_marker in prompt:
            prompt = prompt.replace(example_marker, example_texts)
        else:
            prompt = f"{prompt}\n{example_texts}"


    # 构建完整的 prompt，包括检索的 Examples 和当前的指令
    question = f"{prompt}\n{instruction}"
    print("question:",question)
    messages = []
    messages.append({"role": "user", "content": question})
    answer = llm.request(message=messages)
    messages.append({"role": "assistant", "content": answer})
    print(answer)

    goal_set, priority_act_ls, key_predicates, key_objects = parse_llm_output(answer)

    print("goal",goal_set)
    print("act:",priority_act_ls)
    print("key_predicate",key_predicates)
    print("key_objects:",key_objects)


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

    return goal_set, priority_act_ls, key_predicates, key_objects, messages

def llm_reflect(llm, messages, reflect_prompt):
    messages.append({"role": "user", "content": reflect_prompt})
    answer = llm.request(message=messages)
    messages.append({"role": "assistant", "content": answer})

    print(answer)

    goal_set, priority_act_ls, key_predicates, key_objects = parse_llm_output(answer)

    print("goal",goal_set)
    print("act:",priority_act_ls)
    print("key_predicate",key_predicates)
    print("key_objects:",key_objects)

    return goal_set, priority_act_ls, key_predicates, key_objects, messages
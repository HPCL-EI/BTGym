import pydot
import graphviz


def clear_labels_and_convert_dot_to_svg(dot_file_path, svg_file_path):
    # 读取 .dot 文件
    with open(dot_file_path, 'r') as dot_file:
        dot_data = dot_file.read()

    # 使用 pydot 解析 .dot 文件
    graphs = pydot.graph_from_dot_data(dot_data)

    # 获取图形对象
    graph = graphs[0]

    # 清空所有节点的标签
    for node in graph.get_nodes():
        node.set_label(' ')  # 设置为空格，确保节点仍然可见

    # 清空所有边的标签
    for edge in graph.get_edges():
        edge.set_label(' ')  # 设置为空格，确保边仍然可见

    # 将修改后的图形对象写入 .svg 文件
    graphviz.Source(graph.to_string()).render(svg_file_path, format='svg')


# 示例使用
dot_file_path = 'expanded_bt_obt.dot'
svg_file_path = 'example.svg'
clear_labels_and_convert_dot_to_svg(dot_file_path, svg_file_path)

# 示例使用

# dot_file_path = 'expanded_bt_obt.dot'
# svg_file_path = 'example'
# clear_labels_and_convert_dot_to_svg(dot_file_path, svg_file_path)

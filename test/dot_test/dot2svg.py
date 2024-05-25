import pydot
import graphviz


def convert_dot_to_svg(dot_file_path, svg_file_path):
    # 读取 .dot 文件
    with open(dot_file_path, 'r') as dot_file:
        dot_data = dot_file.read()

    # 使用 pydot 解析 .dot 文件
    graphs = pydot.graph_from_dot_data(dot_data)

    # 获取图形对象
    graph = graphs[0]

    # 将图形对象写入 .svg 文件
    graphviz.Source(graph.to_string()).render(svg_file_path, format='svg')


# 示例使用
dot_file_path = 'easy.dot'
svg_file_path = 'example'
convert_dot_to_svg(dot_file_path, svg_file_path)

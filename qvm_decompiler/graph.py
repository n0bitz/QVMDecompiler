import pydot
import svgutils.transform as sg
from svgutils.compose import Unit

from .ast import ASTNode, ASTVisitor


def graph_svgs(graph, svgs=None, pad=10):
    svgs = svgs or {}
    roots = {name: svg.getroot() for name, svg in svgs.items()}

    # set node sizes based on sizes of svgs they will contain
    for name, svg in svgs.items():
        nodes = graph.get_node(name)
        assert len(nodes) == 1
        node = nodes[0]

        node.set_width((Unit(svg.width).value + pad - 8) / 72)
        node.set_height((Unit(svg.height).value + pad - 8) / 72)

    # compute main graph's layout
    layout = pydot.graph_from_dot_data(graph.create_dot().decode())
    assert len(layout) == 1
    layout = layout[0]

    graph_height = float(layout.get_bb().strip('"').split(",")[3])

    # position svgs according to layout
    for name in svgs:
        nodes = layout.get_node(name)
        assert len(nodes) == 1
        node = nodes[0]

        width = float(node.get_width()) * 72 - pad
        height = float(node.get_height()) * 72 - pad

        x, y = (float(coord) for coord in node.get_pos().strip('"').split(","))
        y = graph_height - y
        y -= height / 2
        x -= width / 2

        roots[name].moveto(x, y)

    # combine everything into a single svg
    svg = sg.fromstring(graph.create_svg().decode())
    svg.append(roots.values())
    return svg


class ASTGraphBuilder(ASTVisitor):
    def __init__(self, graph):
        self.graph = graph
        self.parent = None

    def visit(self, node):
        if not isinstance(node, ASTNode):
            return

        if self.parent is not None:
            self.graph.add_edge(pydot.Edge(id(self.parent), id(node)))

        parent, self.parent = self.parent, node
        label = super().visit(node)
        self.generic_visit(node)
        self.parent = parent

        self.graph.add_node(pydot.Node(id(node), label=label))

    def visit_ASTNode(self, node):
        return type(node).__name__


def graph_function(basic_blocks):
    function_graph = pydot.Dot(
        bgcolor="transparent",
        splines="ortho",
    )
    function_graph.set_node_defaults(shape="box", label="", style="filled", color="gray")
    function_graph.set_edge_defaults(arrowsize=0.75, fontname="monospace")
    function_graph.add_node(pydot.Node(id(basic_blocks[0]), rank="min"))

    block_svgs = {}
    for block in basic_blocks:
        if block is not basic_blocks[0]:
            function_graph.add_node(pydot.Node(id(block)))
        block_graph = pydot.Dot(bgcolor="transparent")
        block_graph.set_node_defaults(shape="box", label="", style="filled", color="lightgray")
        prev_node = None
        node_svgs = {}

        for node in block.nodes:
            block_graph.add_node(pydot.Node(id(node)))
            node_graph = pydot.Dot(bgcolor="transparent", rankdir="LR")
            node_graph.set_node_defaults(fontname="monospace", style="filled", color="white")
            ASTGraphBuilder(node_graph).visit(node)
            node_svgs[f"{id(node)}"] = graph_svgs(node_graph)
            if prev_node is not None:
                block_graph.add_edge(pydot.Edge(id(prev_node), id(node)))
            prev_node = node
        block_svgs[f"{id(block)}"] = graph_svgs(block_graph, node_svgs)

        for succ in block.successors:
            function_graph.add_edge(pydot.Edge(id(block), id(succ), color="darkcyan"))

    return graph_svgs(function_graph, block_svgs)

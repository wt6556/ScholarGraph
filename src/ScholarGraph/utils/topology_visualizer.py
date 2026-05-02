"""拓扑图可视化"""

import os
import logging
import textwrap
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

try:
    from graphviz import Digraph
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    logger.warning("graphviz not installed, will use matplotlib fallback")


def generate_topology_graph(env, output_path="./output/topology.png"):
    """
    生成拓扑图（优先使用 Graphviz，失败则回退到 matplotlib）

    Args:
        env: SharedEnvironment
        output_path: 输出 PNG 路径

    Returns:
        是否成功
    """
    if GRAPHVIZ_AVAILABLE:
        return _generate_graphviz(env, output_path)
    else:
        return _generate_matplotlib(env, output_path)


def _generate_graphviz(env, output_path="./output/topology.png") -> bool:
    """使用 Graphviz 生成拓扑图"""
    try:
        topology = env.topology_store.get_all_topology()
        nodes = topology.get("nodes", [])
        edges = topology.get("edges", [])

        dot = Digraph(format="png")

        # 全局布局优化
        dot.attr(
            rankdir="LR",
            dpi="300",
            size="20,12",
            ratio="compress",
            ranksep="1.2",
            nodesep="0.6"
        )
        dot.attr(fontname="Microsoft YaHei")

        # 节点
        for n in nodes:
            name = n["id"]
            is_root = n.get("is_root", False)

            dot.node(
                name,
                label=f"<<B>{name}</B>>",
                shape="box",
                style="rounded,filled",
                fontname="Microsoft YaHei",
                fontsize="12",
                fillcolor="#FFCDD2" if is_root else "#E3F2FD",
                color="#444444",
                penwidth="1.5",
                margin="0.2,0.1"
            )

        # 语义颜色
        DIRECTION_COLOR = {
            "efficiency": "#1E88E5",
            "generalization": "#43A047",
            "flexibility": "#FB8C00",
            "accuracy": "#E53935",
            "architecture": "#8E24AA",
            "capability": "#00ACC1",
            "system": "#FF7043"
        }
        DEFAULT_COLOR = "#616161"

        # 边
        for e in edges:
            u = e["from"]
            v = e["to"]
            direction = e.get("direction", "")
            desc = e.get("description", "")

            main_dir = direction.split("/")[0] if direction else ""
            color = DIRECTION_COLOR.get(main_dir, DEFAULT_COLOR)

            wrapped_desc = "<BR ALIGN='LEFT'/>".join(
                textwrap.wrap(desc, width=28)
            )

            label = f"""<
            <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                <TR>
                    <TD BGCOLOR="{color}">
                        <FONT COLOR="white" POINT-SIZE="11">
                            <B>{direction}</B>
                        </FONT>
                    </TD>
                </TR>
                <TR>
                    <TD BGCOLOR="#FAFAFA" ALIGN="LEFT">
                        <FONT POINT-SIZE="10">{wrapped_desc}</FONT>
                    </TD>
                </TR>
            </TABLE>
            >"""

            dot.edge(
                u, v,
                label=label,
                color=color,
                penwidth="2.0",
                fontname="Microsoft YaHei"
            )

        # 输出
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dot.render(output_path.replace(".png", ""), cleanup=True)

        logger.info(f"Topology graph saved (Graphviz): {output_path}")
        return True

    except Exception as e:
        logger.error(f"Graphviz failed: {e}")
        return _generate_matplotlib(env, output_path)


def _generate_matplotlib(env, output_path="./output/topology.png") -> bool:
    """使用 matplotlib 生成拓扑图（fallback）"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import networkx as nx
        import numpy as np

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        topology = env.topology_store.get_all_topology()
        edges = topology.get("edges", [])
        nodes = topology.get("nodes", [])

        if not edges and not nodes:
            logger.warning("No topology data to visualize")
            return False

        G = nx.DiGraph()

        all_methods = set()
        method_info = {}
        for node in nodes:
            method_id = node.get("id", "")
            all_methods.add(method_id)
            method_info[method_id] = {
                "is_root": node.get("is_root", False),
            }
        for edge in edges:
            all_methods.add(edge.get("from", ""))
            all_methods.add(edge.get("to", ""))

        for method in all_methods:
            G.add_node(method)

        for edge in edges:
            from_method = edge.get("from", "")
            to_method = edge.get("to", "")
            direction = edge.get("direction", "")
            description = edge.get("description", "")
            if from_method and to_method:
                G.add_edge(from_method, to_method, direction=direction, description=description)

        root_methods = set()
        for node in nodes:
            if node.get("is_root", False):
                root_methods.add(node.get("id", ""))
        if not root_methods:
            in_degree = G.in_degree()
            root_methods = {node for node, deg in in_degree if deg == 0}

        def topo_layout(G, layer_sep=2.5, node_sep=1.8):
            layers = {}
            for node in nx.topological_sort(G):
                if G.in_degree(node) == 0:
                    layers[node] = 0
                else:
                    layers[node] = max(layers[p] for p in G.predecessors(node)) + 1

            layer_nodes = {}
            for node, l in layers.items():
                layer_nodes.setdefault(l, []).append(node)

            pos = {}
            for l, nodes in layer_nodes.items():
                n = len(nodes)
                total_height = (n - 1) * node_sep
                for i, node in enumerate(nodes):
                    y = i * node_sep - total_height / 2
                    x = l * layer_sep
                    pos[node] = (x, y)
            return pos

        pos = topo_layout(G)

        xs = np.array([p[0] for p in pos.values()])
        ys = np.array([p[1] for p in pos.values()])
        x_range = xs.max() - xs.min() if xs.max() != xs.min() else 1
        y_range = ys.max() - ys.min() if ys.max() != ys.min() else 1

        margin = 1.5
        scale_x = (10 - 2 * margin) / x_range if x_range > 0 else 1
        scale_y = (8 - 2 * margin) / y_range if y_range > 0 else 1
        scale = min(scale_x, scale_y)

        offset_x, offset_y = xs.min(), ys.min()
        pos = {
            node: ((x - offset_x) * scale + margin, (y - offset_y) * scale + margin)
            for node, (x, y) in pos.items()
        }

        _, ax = plt.subplots(1, 1, figsize=(22, 16))

        root_color = '#FF6B6B'
        level1_color = '#4ECDC4'
        level2_color = '#45B7D1'
        level3_color = '#96CEB4'
        default_color = '#DDA0DD'

        def get_node_color(node):
            if node in root_methods:
                return root_color
            try:
                min_dist = float('inf')
                for root in root_methods:
                    if nx.has_path(G, root, node):
                        dist = nx.shortest_path_length(G, root, node)
                        min_dist = min(min_dist, dist)
                if min_dist == 1:
                    return level1_color
                elif min_dist == 2:
                    return level2_color
                elif min_dist <= 3:
                    return level3_color
                return default_color
            except:
                return default_color

        for edge in G.edges():
            start = pos[edge[0]]
            end = pos[edge[1]]
            direction = G.edges[edge].get('direction', '')
            description = G.edges[edge].get('description', '')

            path = mpatches.FancyArrowPatch(
                start, end,
                connectionstyle="arc3,rad=0.1",
                arrowstyle='-|>',
                mutation_scale=18,
                color='#888888',
                linewidth=1.5,
                alpha=0.8
            )
            ax.add_patch(path)

            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2

            if direction:
                dir_short = direction.split('/')[-1] if '/' in direction else direction
                ax.annotate(f"[{dir_short}]",
                          xy=(mid_x, mid_y + 0.3),
                          fontsize=6, color='#D32F2F', fontweight='bold',
                          ha='center', va='bottom', alpha=0.9)

            if description:
                desc_short = description[:35] + "..." if len(description) > 35 else description
                ax.annotate(desc_short.replace('\n', ' '),
                          xy=(mid_x, mid_y - 0.3),
                          fontsize=5, color='#555555',
                          ha='center', va='top', alpha=0.75)

        for node in G.nodes():
            x, y = pos[node]
            color = get_node_color(node)
            radius = 0.35 if node in root_methods else 0.25
            circle = plt.Circle((x, y), radius, color=color, ec='#333', linewidth=2, zorder=3)
            ax.add_patch(circle)

            label = node if len(node) <= 25 else node[:23] + "..."
            fontsize = 10 if node in root_methods else 9
            ax.annotate(label, (x, y),
                       fontsize=fontsize,
                       fontweight='bold' if node in root_methods else 'normal',
                       ha='center', va='center', color='#1a1a1a', zorder=4)

        ax.set_title("LoRA-related Methods Topology", fontsize=18, fontweight='bold', pad=20)

        legend_elements = [
            mpatches.Patch(facecolor=root_color, edgecolor='#333', label='Root Method'),
            mpatches.Patch(facecolor=level1_color, edgecolor='#333', label='Direct Improvement'),
            mpatches.Patch(facecolor=level2_color, edgecolor='#333', label='2nd Level'),
            mpatches.Patch(facecolor=level3_color, edgecolor='#333', label='3rd Level'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        ax.set_xlim(-1.5, max(p[0] for p in pos.values()) + 1.5)
        ax.set_ylim(min(p[1] for p in pos.values()) - 1, max(p[1] for p in pos.values()) + 1)
        ax.axis('off')
        ax.set_aspect('equal')
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                   facecolor="white", edgecolor="none")
        plt.close()

        logger.info(f"Topology graph saved (matplotlib): {output_path}")
        return True

    except Exception as e:
        logger.error(f"matplotlib fallback failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_topology_json(env, output_path: str = "./output/topology.json") -> bool:
    """导出拓扑数据为 JSON"""
    try:
        topology = env.topology_store.get_all_topology()

        import json
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(topology, f, indent=2, ensure_ascii=False)

        logger.info(f"Topology JSON exported to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to export topology JSON: {e}")
        return False
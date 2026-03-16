"""Hydraulic graph analysis functions wrapping topologicpy Graph methods."""
from __future__ import annotations

import math
import logging
import os
from typing import Any

import networkx as nx
import numpy as np

from .bridge import bridge_to_topologicpy, bridge_to_networkx, _require_topologicpy

logger = logging.getLogger(__name__)


def _compute_flow_capacity(
    diameter_mm: float | None,
    reference_velocity_mps: float = 3.0,
) -> float:
    """
    Compute maximum flow capacity in litres per second from pipe diameter.
    Q = v * pi * (D/2)^2  where D is in metres.
    Returns 0.0 if diameter is None or <= 0.
    """
    if diameter_mm is None or diameter_mm <= 0:
        return 0.0
    d_m = diameter_mm / 1000.0
    area = math.pi * (d_m / 2.0) ** 2
    q_m3s = reference_velocity_mps * area
    return q_m3s * 1000.0  # convert m^3/s to L/s


def _compute_hydraulic_resistance(
    length_m: float,
    diameter_mm: float | None,
    friction_factor: float = 0.02,
) -> float:
    """
    Compute Darcy-Weisbach hydraulic resistance.
    R = (f * L) / (D * A^2)
    where A = pi * (D/2)^2, D in metres, L in metres.
    Returns float('inf') if diameter is None or <= 0 or length <= 0.
    """
    if diameter_mm is None or diameter_mm <= 0:
        return float("inf")
    if length_m is None or length_m <= 0:
        return float("inf")
    d_m = diameter_mm / 1000.0
    area = math.pi * (d_m / 2.0) ** 2
    return (friction_factor * length_m) / (d_m * area ** 2)


def _find_vertex_by_global_id(tp_graph: Any, global_id: str) -> Any:
    """Find a topologicpy Vertex by its global_id dictionary value."""
    _, _, Graph, Dictionary, TPTopology = _require_topologicpy()
    for v in Graph.Vertices(tp_graph):
        d = TPTopology.Dictionary(v)
        if d:
            attrs = Dictionary.PythonDictionary(d)
            if attrs.get("global_id") == global_id:
                return v
    return None


def _annotate_edges_with_capacity(tp_graph: Any, capacity_key: str = "flow_capacity_lps") -> None:
    """Compute and set flow_capacity_lps on each edge Dictionary."""
    _, _, Graph, Dictionary, TPTopology = _require_topologicpy()
    for e in Graph.Edges(tp_graph):
        d = TPTopology.Dictionary(e)
        attrs = Dictionary.PythonDictionary(d) if d else {}
        diam = attrs.get("nominal_diameter_mm")
        if isinstance(diam, str):
            try:
                diam = float(diam)
            except ValueError:
                diam = None
        cap = _compute_flow_capacity(diam)
        keys = list(attrs.keys()) + [capacity_key]
        values = list(attrs.values()) + [cap]
        new_d = Dictionary.ByKeysValues(keys, values)
        TPTopology.SetDictionary(e, new_d)


def _annotate_edges_with_resistance(
    tp_graph: Any, resistance_key: str = "hydraulic_resistance"
) -> None:
    """Compute and set hydraulic_resistance on each edge Dictionary."""
    _, _, Graph, Dictionary, TPTopology = _require_topologicpy()
    for e in Graph.Edges(tp_graph):
        d = TPTopology.Dictionary(e)
        attrs = Dictionary.PythonDictionary(d) if d else {}
        length = attrs.get("length_m", 0.0)
        diam = attrs.get("nominal_diameter_mm")
        if isinstance(length, str):
            try:
                length = float(length)
            except ValueError:
                length = 0.0
        if isinstance(diam, str):
            try:
                diam = float(diam)
            except ValueError:
                diam = None
        res = _compute_hydraulic_resistance(length, diam)
        # Cap inf to a large finite number for graph algorithms
        if math.isinf(res):
            res = 1e12
        keys = list(attrs.keys()) + [resistance_key]
        values = list(attrs.values()) + [res]
        new_d = Dictionary.ByKeysValues(keys, values)
        TPTopology.SetDictionary(e, new_d)


def critical_pipe_by_flow_capacity(
    tp_graph: Any,
    source_global_id: str,
    sink_global_id: str,
    capacity_key: str = "flow_capacity_lps",
) -> dict:
    """
    Identify the bottleneck pipe segment using maximum flow analysis.

    Steps:
    1. Compute flow_capacity_lps for each edge if not already present
    2. Find source and sink Vertices by matching global_id in Vertex Dictionary
    3. Try Graph.MaximumFlow or fall back to networkx
    4. Extract max flow value and identify the saturated (critical) edge

    Returns dict with keys:
        'max_flow_lps': float
        'critical_edge_global_ids': list of (src_id, tgt_id) tuples
        'flow_values': dict mapping edge pairs to their flow values
    """
    _, _, Graph, Dictionary, TPTopology = _require_topologicpy()

    _annotate_edges_with_capacity(tp_graph, capacity_key)

    # Try topologicpy MaximumFlow first
    source_v = _find_vertex_by_global_id(tp_graph, source_global_id)
    sink_v = _find_vertex_by_global_id(tp_graph, sink_global_id)

    if source_v is None:
        raise ValueError(f"Source vertex with global_id '{source_global_id}' not found.")
    if sink_v is None:
        raise ValueError(f"Sink vertex with global_id '{sink_global_id}' not found.")

    try:
        mf_result = Graph.MaximumFlow(
            tp_graph, source_v, sink_v, edgeKeyFwd=capacity_key
        )
        if mf_result is not None and isinstance(mf_result, (int, float)):
            max_flow = float(mf_result)
        elif mf_result is not None and isinstance(mf_result, (list, tuple)):
            max_flow = float(mf_result[0]) if mf_result else 0.0
        else:
            raise RuntimeError("Unexpected MaximumFlow result")
    except Exception as e:
        logger.info("topologicpy MaximumFlow failed (%s), falling back to networkx.", e)
        # Fallback to networkx
        nx_g = bridge_to_networkx(tp_graph)
        # Set capacity on edges
        for u, v, data in nx_g.edges(data=True):
            diam = data.get("nominal_diameter_mm")
            if isinstance(diam, str):
                try:
                    diam = float(diam)
                except ValueError:
                    diam = None
            data[capacity_key] = _compute_flow_capacity(diam)

        try:
            max_flow_val, flow_dict = nx.maximum_flow(
                nx_g, source_global_id, sink_global_id, capacity=capacity_key
            )
        except nx.NetworkXError:
            # If no path exists, return zero flow
            return {
                "max_flow_lps": 0.0,
                "critical_edge_global_ids": [],
                "flow_values": {},
            }

        max_flow = float(max_flow_val)

        # Find critical (saturated) edges
        flow_values = {}
        critical_edges = []
        for u, targets in flow_dict.items():
            for v, flow in targets.items():
                if flow > 0:
                    flow_values[(u, v)] = flow
                    cap = nx_g.edges[u, v].get(capacity_key, float("inf"))
                    if cap > 0 and abs(flow - cap) < 1e-6:
                        critical_edges.append((u, v))

        return {
            "max_flow_lps": max_flow,
            "critical_edge_global_ids": critical_edges,
            "flow_values": flow_values,
        }

    # If topologicpy MaximumFlow succeeded, find critical edges via nx fallback
    # since we need flow decomposition
    nx_g = bridge_to_networkx(tp_graph)
    for u, v, data in nx_g.edges(data=True):
        diam = data.get("nominal_diameter_mm")
        if isinstance(diam, str):
            try:
                diam = float(diam)
            except ValueError:
                diam = None
        data[capacity_key] = _compute_flow_capacity(diam)

    try:
        _, flow_dict = nx.maximum_flow(
            nx_g, source_global_id, sink_global_id, capacity=capacity_key
        )
    except nx.NetworkXError:
        flow_dict = {}

    flow_values = {}
    critical_edges = []
    for u, targets in flow_dict.items():
        for v, flow in targets.items():
            if flow > 0:
                flow_values[(u, v)] = flow
                cap = nx_g.edges[u, v].get(capacity_key, float("inf"))
                if cap > 0 and abs(flow - cap) < 1e-6:
                    critical_edges.append((u, v))

    return {
        "max_flow_lps": max_flow,
        "critical_edge_global_ids": critical_edges,
        "flow_values": flow_values,
    }


def hydraulic_resistance_shortest_path(
    tp_graph: Any,
    source_global_id: str,
    terminal_global_ids: list[str],
    resistance_key: str = "hydraulic_resistance",
) -> dict:
    """
    Find the least-resistance hydraulic path from source to each terminal.

    Uses networkx dijkstra_path with hydraulic_resistance as weight.

    Returns dict mapping terminal_global_id -> {
        'path_global_ids': list[str],
        'total_resistance': float,
        'path_length_m': float,
        'min_diameter_mm': float
    }
    """
    _annotate_edges_with_resistance(tp_graph, resistance_key)

    # Use networkx for reliable weighted shortest path
    nx_g = bridge_to_networkx(tp_graph)

    # Set resistance weight on edges
    for u, v, data in nx_g.edges(data=True):
        length = data.get("length_m", 0.0)
        diam = data.get("nominal_diameter_mm")
        if isinstance(length, str):
            try:
                length = float(length)
            except ValueError:
                length = 0.0
        if isinstance(diam, str):
            try:
                diam = float(diam)
            except ValueError:
                diam = None
        res = _compute_hydraulic_resistance(length, diam)
        if math.isinf(res):
            res = 1e12
        data[resistance_key] = res

    results = {}
    for term_id in terminal_global_ids:
        try:
            path = nx.dijkstra_path(nx_g, source_global_id, term_id, weight=resistance_key)
            total_res = 0.0
            total_length = 0.0
            min_diam = float("inf")

            for i in range(len(path) - 1):
                edge_data = nx_g.edges[path[i], path[i + 1]]
                total_res += edge_data.get(resistance_key, 0.0)
                length = edge_data.get("length_m", 0.0)
                if isinstance(length, str):
                    length = float(length)
                total_length += length
                diam = edge_data.get("nominal_diameter_mm")
                if isinstance(diam, str):
                    try:
                        diam = float(diam)
                    except ValueError:
                        diam = None
                if diam is not None and diam > 0:
                    min_diam = min(min_diam, diam)

            if math.isinf(min_diam):
                min_diam = 0.0

            results[term_id] = {
                "path_global_ids": path,
                "total_resistance": total_res,
                "path_length_m": total_length,
                "min_diameter_mm": min_diam,
            }
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            logger.warning("No path from %s to %s.", source_global_id, term_id)

    return results


def pipe_segment_centrality(
    tp_graph: Any,
    weight_key: str = "length_m",
) -> dict:
    """
    Compute betweenness centrality for each node.

    Falls back to nx.betweenness_centrality() for reliability.

    Returns dict mapping global_id -> float (normalized centrality).
    """
    _, _, Graph, Dictionary, TPTopology = _require_topologicpy()

    # Try topologicpy first
    try:
        result = Graph.BetweennessCentrality(tp_graph, normalize=True)
        if result is not None:
            # BetweennessCentrality may return the graph with annotated vertices
            # Read centrality values from vertex dictionaries
            centrality = {}
            vertices = Graph.Vertices(result if result is not None else tp_graph)
            for v in vertices:
                d = TPTopology.Dictionary(v)
                attrs = Dictionary.PythonDictionary(d) if d else {}
                gid = attrs.get("global_id", "unknown")
                bc = attrs.get("betweenness_centrality", 0.0)
                if isinstance(bc, str):
                    try:
                        bc = float(bc)
                    except ValueError:
                        bc = 0.0
                centrality[gid] = bc

            if centrality and any(v != 0.0 for v in centrality.values()):
                return centrality
    except Exception as e:
        logger.info("topologicpy BetweennessCentrality failed (%s), using networkx.", e)

    # Fallback to networkx
    nx_g = bridge_to_networkx(tp_graph)
    bc = nx.betweenness_centrality(nx_g, normalized=True)
    return bc


def export_annotated_graph_html(
    tp_graph: Any,
    output_path: str,
    color_key: str = "bc_color",
    label_key: str = "ifc_type",
    height: int = 900,
) -> str:
    """
    Export the graph as an interactive HTML visualization.

    Tries Graph.PyvisGraph() first. If unavailable, falls back to
    building a pyvis Network manually from the bridge_to_networkx() output.

    Returns the absolute path to the written HTML file.
    """
    _, _, Graph, _, _ = _require_topologicpy()

    abs_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_path) if os.path.dirname(abs_path) else ".", exist_ok=True)

    try:
        result = Graph.PyvisGraph(
            tp_graph,
            abs_path,
            height=height,
            vertexLabelKey=label_key,
            vertexColorKey=color_key,
        )
        if result is not None and os.path.exists(abs_path):
            return abs_path
    except Exception as e:
        logger.info("PyvisGraph failed (%s), falling back to manual pyvis.", e)

    # Fallback: manual HTML generation
    nx_g = bridge_to_networkx(tp_graph)

    html_content = ["<!DOCTYPE html><html><head><title>Graph</title></head><body>"]
    html_content.append(f"<h2>Graph Visualization ({nx_g.number_of_nodes()} nodes, "
                        f"{nx_g.number_of_edges()} edges)</h2>")
    html_content.append("<pre>")
    for node, data in nx_g.nodes(data=True):
        html_content.append(f"Node: {data.get(label_key, node)}, id={node}")
    for u, v, data in nx_g.edges(data=True):
        html_content.append(f"Edge: {u} -> {v}, length_m={data.get('length_m', '?')}")
    html_content.append("</pre></body></html>")

    with open(abs_path, "w") as f:
        f.write("\n".join(html_content))

    return abs_path

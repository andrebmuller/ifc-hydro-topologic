"""Bidirectional bridge between ifc-hydro networkx graphs and topologicpy Graphs."""
from __future__ import annotations

import logging
import warnings
from typing import Any

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# Required node attributes after enrichment
_REQUIRED_NODE_ATTRS = {"x", "y", "z", "ifc_type", "global_id"}
# Default edge attributes
_DEFAULT_EDGE_ATTRS = {"length_m": 0.0, "fitting_loss_factor": 0.0, "nominal_diameter_mm": None}


def _require_topologicpy():
    """Import and return topologicpy modules, raising ImportError if unavailable."""
    try:
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Graph import Graph
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology as TPTopology
        return Vertex, Edge, Graph, Dictionary, TPTopology
    except ImportError as e:
        raise ImportError(
            "topologicpy is required for bridge operations. "
            "Install with: pip install topologicpy>=0.9.7"
        ) from e


def enrich_graph(
    nx_graph: nx.DiGraph,
    ifc_model: Any | None = None,
) -> nx.DiGraph:
    """
    Walk the networkx graph and ensure all nodes have the required attribute
    schema for bridge_to_topologicpy().

    If ifc_model is provided and node attributes are missing, attempts to
    extract them from the IFC model using ifcopenshell.

    If ifc_model is None (synthetic graph), validates that required attributes
    exist and fills defaults where appropriate.

    Required node attributes after enrichment:
        x, y, z (float) -- 3D coordinates in metres
        ifc_type (str)
        global_id (str)

    Modifies nx_graph in-place. Returns the same object.
    """
    for node_id, data in nx_graph.nodes(data=True):
        # Ensure global_id
        if "global_id" not in data:
            data["global_id"] = str(node_id)

        # Ensure ifc_type
        if "ifc_type" not in data:
            data["ifc_type"] = "Unknown"

        # Handle coordinates
        has_coords = all(k in data and data[k] is not None for k in ("x", "y", "z"))
        if not has_coords:
            if ifc_model is not None:
                try:
                    from ifc_hydro.core.geom import Geom
                    element = ifc_model.by_guid(data.get("global_id", str(node_id)))
                    center = Geom.get_bbox_center(element)
                    data.setdefault("x", center[0])
                    data.setdefault("y", center[1])
                    data.setdefault("z", center[2])
                except Exception:
                    logger.warning(
                        "Could not extract coordinates from IFC model for node %s. "
                        "Defaulting to (0, 0, 0).",
                        node_id,
                    )
                    data.setdefault("x", 0.0)
                    data.setdefault("y", 0.0)
                    data.setdefault("z", 0.0)
            else:
                logger.warning(
                    "Node %s is missing x/y/z coordinates and no ifc_model provided. "
                    "Defaulting to (0, 0, 0).",
                    node_id,
                )
                data.setdefault("x", 0.0)
                data.setdefault("y", 0.0)
                data.setdefault("z", 0.0)

    # Fill default edge attributes
    for u, v, data in nx_graph.edges(data=True):
        for attr, default in _DEFAULT_EDGE_ATTRS.items():
            data.setdefault(attr, default)

    return nx_graph


def bridge_to_topologicpy(
    nx_graph: nx.DiGraph,
    ifc_model: Any | None = None,
    enrich: bool = True,
) -> Any:
    """
    Convert an ifc-hydro networkx DiGraph into a topologicpy Graph.

    Steps:
    1. If enrich=True, call enrich_graph()
    2. For each node, create a topologicpy Vertex at (x, y, z)
    3. Attach all node attributes as a Vertex Dictionary
    4. For each edge, create a topologicpy Edge between the corresponding Vertices
    5. Attach all edge attributes as an Edge Dictionary
    6. Build the Graph from Vertices and Edges using Graph.ByVerticesEdges()
    7. Validate: Graph.Order() == nx_graph.number_of_nodes()

    Coordinate convention: topologicpy works in metres. If coordinates appear
    to be in millimetres (any coordinate > 1000), log a warning suggesting
    unit conversion. Do NOT auto-convert -- the caller is responsible for units.

    Returns topologic_core.Graph.
    Raises ValueError if any node is missing x, y, z after enrichment.
    Raises ImportError if topologicpy is not installed.
    """
    Vertex, Edge, Graph, Dictionary, TPTopology = _require_topologicpy()

    if enrich:
        enrich_graph(nx_graph, ifc_model)

    # Validate coordinates exist
    for node_id, data in nx_graph.nodes(data=True):
        if any(data.get(k) is None for k in ("x", "y", "z")):
            raise ValueError(
                f"Node {node_id} is missing x/y/z coordinates. "
                "Call enrich_graph() first or set enrich=True."
            )

    # Check for large coordinates (likely mm instead of m)
    warned_mm = False
    for node_id, data in nx_graph.nodes(data=True):
        if not warned_mm and any(abs(data[k]) > 1000 for k in ("x", "y", "z")):
            logger.warning(
                "Coordinates > 1000 detected (node %s). "
                "topologicpy works in metres — are your coordinates in millimetres?",
                node_id,
            )
            warned_mm = True

    # Build vertex mapping, ensuring unique coordinates
    gid_to_vertex: dict[str, Any] = {}
    seen_coords: dict[tuple[float, float, float], str] = {}
    vertices = []

    for node_id, data in nx_graph.nodes(data=True):
        gid = data["global_id"]
        x, y, z = float(data["x"]), float(data["y"]), float(data["z"])
        coord_key = (round(x, 4), round(y, 4), round(z, 4))

        if coord_key in seen_coords:
            # Add tiny jitter to avoid topologicpy merging vertices
            logger.warning(
                "Duplicate coordinates detected for nodes %s and %s at (%s, %s, %s). "
                "Adding jitter to prevent vertex merging.",
                seen_coords[coord_key], gid, x, y, z,
            )
            x += np.random.default_rng(hash(gid) % 2**32).uniform(1e-6, 1e-4)
            y += np.random.default_rng(hash(gid) % 2**32 + 1).uniform(1e-6, 1e-4)

        seen_coords[(round(x, 4), round(y, 4), round(z, 4))] = gid

        vertex = Vertex.ByCoordinates(x, y, z)

        # Build dictionary from all node attributes
        keys = []
        values = []
        for k, v in data.items():
            if v is None:
                continue
            keys.append(str(k))
            if isinstance(v, (int, float)):
                values.append(float(v))
            else:
                values.append(str(v))

        if keys:
            d = Dictionary.ByKeysValues(keys, values)
            TPTopology.SetDictionary(vertex, d)

        gid_to_vertex[gid] = vertex
        vertices.append(vertex)

    # Build edges
    edges = []
    for u, v, data in nx_graph.edges(data=True):
        u_gid = nx_graph.nodes[u]["global_id"]
        v_gid = nx_graph.nodes[v]["global_id"]

        v_src = gid_to_vertex.get(u_gid)
        v_tgt = gid_to_vertex.get(v_gid)

        if v_src is None or v_tgt is None:
            logger.warning("Skipping edge %s -> %s: vertex not found.", u_gid, v_gid)
            continue

        edge = Edge.ByVertices([v_src, v_tgt])
        if edge is None:
            logger.warning(
                "Could not create edge between %s and %s (possibly same coordinates).",
                u_gid, v_gid,
            )
            continue

        # Build edge dictionary with direction info
        keys = ["_src_id", "_tgt_id"]
        values: list[Any] = [str(u_gid), str(v_gid)]
        for k, val in data.items():
            if val is None:
                continue
            keys.append(str(k))
            if isinstance(val, (int, float)):
                values.append(float(val))
            else:
                values.append(str(val))

        if keys:
            d = Dictionary.ByKeysValues(keys, values)
            TPTopology.SetDictionary(edge, d)

        edges.append(edge)

    # Build graph
    tp_graph = Graph.ByVerticesEdges(vertices, edges)
    if tp_graph is None:
        raise RuntimeError("Failed to create topologicpy Graph from vertices and edges.")

    # Validate
    order = Graph.Order(tp_graph)
    expected = nx_graph.number_of_nodes()
    if order != expected:
        logger.warning(
            "Graph order mismatch: topologicpy has %d vertices, networkx has %d nodes. "
            "Possible vertex merging due to close coordinates.",
            order, expected,
        )

    return tp_graph


def bridge_to_networkx(
    tp_graph: Any,
) -> nx.DiGraph:
    """
    Convert a topologicpy Graph back to a networkx DiGraph.

    Steps:
    1. Extract all Vertices from the Graph via Graph.Vertices()
    2. For each Vertex, read its Dictionary and extract all key-value pairs
    3. Use 'global_id' from the Dictionary as the networkx node identifier
    4. Extract all Edges via Graph.Edges()
    5. For each Edge, read its Dictionary, identify source and target nodes
    6. If directed: use _src_id/_tgt_id from edge Dictionary
       If undirected: use Edge start/end vertices and read their global_id
    7. Add edges with all edge Dictionary contents as edge attributes

    Returns nx.DiGraph with all attributes preserved.
    """
    Vertex, Edge, Graph, Dictionary, TPTopology = _require_topologicpy()

    G = nx.DiGraph()

    # Extract vertices
    tp_vertices = Graph.Vertices(tp_graph)
    coord_to_gid: dict[tuple[float, float, float], str] = {}

    for v in tp_vertices:
        d = TPTopology.Dictionary(v)
        attrs = Dictionary.PythonDictionary(d) if d else {}

        gid = attrs.get("global_id", f"v_{Vertex.X(v)}_{Vertex.Y(v)}_{Vertex.Z(v)}")

        # Restore x/y/z from vertex geometry if not in dict
        attrs.setdefault("x", Vertex.X(v))
        attrs.setdefault("y", Vertex.Y(v))
        attrs.setdefault("z", Vertex.Z(v))

        # Convert numeric strings back to appropriate types
        cleaned = {}
        for k, val in attrs.items():
            if isinstance(val, str):
                try:
                    cleaned[k] = float(val)
                except ValueError:
                    cleaned[k] = val
            else:
                cleaned[k] = val

        G.add_node(gid, **cleaned)
        coord_key = (round(Vertex.X(v), 4), round(Vertex.Y(v), 4), round(Vertex.Z(v), 4))
        coord_to_gid[coord_key] = gid

    # Extract edges
    tp_edges = Graph.Edges(tp_graph)

    for e in tp_edges:
        d = TPTopology.Dictionary(e)
        attrs = Dictionary.PythonDictionary(d) if d else {}

        src_id = attrs.pop("_src_id", None)
        tgt_id = attrs.pop("_tgt_id", None)

        if src_id is None or tgt_id is None:
            # Fall back to vertex coordinates
            sv = Edge.StartVertex(e)
            ev = Edge.EndVertex(e)
            sv_key = (round(Vertex.X(sv), 4), round(Vertex.Y(sv), 4), round(Vertex.Z(sv), 4))
            ev_key = (round(Vertex.X(ev), 4), round(Vertex.Y(ev), 4), round(Vertex.Z(ev), 4))
            src_id = coord_to_gid.get(sv_key, str(sv_key))
            tgt_id = coord_to_gid.get(ev_key, str(ev_key))

        # Convert numeric strings
        cleaned = {}
        for k, val in attrs.items():
            if isinstance(val, str):
                try:
                    cleaned[k] = float(val)
                except ValueError:
                    cleaned[k] = val
            else:
                cleaned[k] = val

        G.add_edge(src_id, tgt_id, **cleaned)

    return G

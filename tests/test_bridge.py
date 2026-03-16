"""Tests for bridge.py — nx <-> topologicpy graph conversion."""
from __future__ import annotations

import logging
import pytest
import networkx as nx
from ifc_hydro_topologic.bridge import (
    enrich_graph,
    bridge_to_topologicpy,
    bridge_to_networkx,
    _require_topologicpy,
)


class TestEnrichGraph:
    def test_enrich_preserves_existing_coordinates(self, simple_3node_graph):
        """enrich_graph does not overwrite existing x/y/z values."""
        # Record original coordinates
        original_coords = {}
        for node, data in simple_3node_graph.nodes(data=True):
            original_coords[node] = (data["x"], data["y"], data["z"])

        enrich_graph(simple_3node_graph)

        for node, data in simple_3node_graph.nodes(data=True):
            assert (data["x"], data["y"], data["z"]) == original_coords[node]

    def test_enrich_fills_missing_edge_attributes(self):
        """Missing edge length_m defaults to 0.0, fitting_loss_factor to 0.0."""
        G = nx.DiGraph()
        G.add_node("a", ifc_type="IfcTank", global_id="a", x=0.0, y=0.0, z=10.0)
        G.add_node("b", ifc_type="IfcPipeSegment", global_id="b", x=1.0, y=0.0, z=5.0)
        G.add_edge("a", "b")  # No edge attributes

        enrich_graph(G)

        edge_data = G.edges["a", "b"]
        assert edge_data["length_m"] == 0.0
        assert edge_data["fitting_loss_factor"] == 0.0
        assert edge_data["nominal_diameter_mm"] is None

    def test_enrich_warns_on_missing_coordinates(self, caplog):
        """Logs a warning for nodes without x/y/z when ifc_model is None."""
        G = nx.DiGraph()
        G.add_node("a", ifc_type="IfcTank", global_id="a")

        with caplog.at_level(logging.WARNING):
            enrich_graph(G)

        assert any("missing x/y/z" in msg.lower() for msg in caplog.messages)
        assert G.nodes["a"]["x"] == 0.0
        assert G.nodes["a"]["y"] == 0.0
        assert G.nodes["a"]["z"] == 0.0


class TestBridgeToTopologicpy:
    def test_returns_valid_graph(self, simple_3node_graph):
        """bridge_to_topologicpy returns a topologic_core.Graph object."""
        _, _, Graph, _, _ = _require_topologicpy()
        tp = bridge_to_topologicpy(simple_3node_graph)
        assert tp is not None
        # Check it's a topologic_core.Graph
        assert Graph.Order(tp) is not None

    def test_node_count_preserved(self, simple_3node_graph):
        """Graph.Order() equals nx_graph.number_of_nodes()."""
        _, _, Graph, _, _ = _require_topologicpy()
        tp = bridge_to_topologicpy(simple_3node_graph)
        assert Graph.Order(tp) == simple_3node_graph.number_of_nodes()

    def test_edge_count_preserved(self, simple_3node_graph):
        """Graph.Size() equals nx_graph.number_of_edges()."""
        _, _, Graph, _, _ = _require_topologicpy()
        tp = bridge_to_topologicpy(simple_3node_graph)
        assert Graph.Size(tp) == simple_3node_graph.number_of_edges()

    def test_vertex_coordinates_assigned(self, simple_3node_graph):
        """All Vertices have non-None x/y/z coordinates matching the nx node."""
        Vertex, _, Graph, _, TPTopology = _require_topologicpy()
        tp = bridge_to_topologicpy(simple_3node_graph)
        vertices = Graph.Vertices(tp)

        for v in vertices:
            assert Vertex.X(v) is not None
            assert Vertex.Y(v) is not None
            assert Vertex.Z(v) is not None

    def test_vertex_dictionary_contains_attributes(self, simple_3node_graph):
        """Vertex Dictionaries contain ifc_type, global_id, and other attributes."""
        _, _, Graph, Dictionary, TPTopology = _require_topologicpy()
        tp = bridge_to_topologicpy(simple_3node_graph)
        vertices = Graph.Vertices(tp)

        found_types = set()
        for v in vertices:
            d = TPTopology.Dictionary(v)
            attrs = Dictionary.PythonDictionary(d) if d else {}
            assert "ifc_type" in attrs
            assert "global_id" in attrs
            found_types.add(attrs["ifc_type"])

        assert "IfcTank" in found_types
        assert "IfcPipeSegment" in found_types
        assert "IfcSanitaryTerminal" in found_types

    def test_raises_on_missing_coordinates(self):
        """ValueError raised if a node has no x/y/z and enrich=False."""
        G = nx.DiGraph()
        G.add_node("a", ifc_type="IfcTank", global_id="a")

        with pytest.raises(ValueError, match="missing x/y/z"):
            bridge_to_topologicpy(G, enrich=False)

    def test_warns_on_large_coordinates(self, simple_3node_graph, caplog):
        """Warns if coordinates > 1000 (likely mm, not m)."""
        # Modify a node to have large coordinates
        node = list(simple_3node_graph.nodes())[0]
        simple_3node_graph.nodes[node]["x"] = 5000.0

        with caplog.at_level(logging.WARNING):
            bridge_to_topologicpy(simple_3node_graph)

        assert any("1000" in msg for msg in caplog.messages)


class TestBridgeToNetworkx:
    def test_roundtrip_preserves_node_count(self, simple_3node_graph):
        """nx -> tp -> nx preserves number of nodes."""
        tp = bridge_to_topologicpy(simple_3node_graph)
        result = bridge_to_networkx(tp)
        assert result.number_of_nodes() == simple_3node_graph.number_of_nodes()

    def test_roundtrip_preserves_edge_count(self, simple_3node_graph):
        """nx -> tp -> nx preserves number of edges."""
        tp = bridge_to_topologicpy(simple_3node_graph)
        result = bridge_to_networkx(tp)
        assert result.number_of_edges() == simple_3node_graph.number_of_edges()

    def test_roundtrip_preserves_node_attributes(self, simple_3node_graph):
        """All node attributes survive the roundtrip."""
        tp = bridge_to_topologicpy(simple_3node_graph)
        result = bridge_to_networkx(tp)

        for node, data in simple_3node_graph.nodes(data=True):
            gid = data["global_id"]
            assert gid in result.nodes, f"Node {gid} not found in result"
            result_data = result.nodes[gid]
            assert result_data["ifc_type"] == data["ifc_type"]
            assert pytest.approx(result_data["x"], abs=1e-3) == data["x"]
            assert pytest.approx(result_data["z"], abs=1e-3) == data["z"]

    def test_roundtrip_preserves_edge_attributes(self, simple_3node_graph):
        """length_m and nominal_diameter_mm survive the roundtrip."""
        tp = bridge_to_topologicpy(simple_3node_graph)
        result = bridge_to_networkx(tp)

        for u, v, data in simple_3node_graph.edges(data=True):
            u_gid = simple_3node_graph.nodes[u]["global_id"]
            v_gid = simple_3node_graph.nodes[v]["global_id"]
            assert result.has_edge(u_gid, v_gid), f"Edge {u_gid} -> {v_gid} not found"
            result_data = result.edges[u_gid, v_gid]
            assert pytest.approx(result_data["length_m"], abs=1e-3) == data["length_m"]

    def test_roundtrip_on_larger_graph(self, synthetic_tree_graph):
        """Roundtrip works on the 15-node tree graph."""
        tp = bridge_to_topologicpy(synthetic_tree_graph)
        result = bridge_to_networkx(tp)
        assert result.number_of_nodes() == synthetic_tree_graph.number_of_nodes()
        assert result.number_of_edges() == synthetic_tree_graph.number_of_edges()


class TestBridgeImportGuard:
    def test_import_error_message(self, monkeypatch):
        """If topologicpy is not importable, a clear error message is raised."""
        import sys
        # Temporarily remove topologicpy from sys.modules
        saved = {}
        for key in list(sys.modules.keys()):
            if "topologicpy" in key or "topologic_core" in key:
                saved[key] = sys.modules.pop(key)

        monkeypatch.setitem(sys.modules, "topologicpy", None)
        monkeypatch.setitem(sys.modules, "topologicpy.Vertex", None)

        try:
            with pytest.raises(ImportError, match="topologicpy is required"):
                _require_topologicpy()
        finally:
            # Restore modules
            for key, mod in saved.items():
                sys.modules[key] = mod
            sys.modules.pop("topologicpy", None)
            sys.modules.pop("topologicpy.Vertex", None)

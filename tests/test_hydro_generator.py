"""Tests for hydro_generator.py — synthetic hydraulic network generation."""
from __future__ import annotations

import numpy as np
import networkx as nx
import pytest
from ifc_hydro_topologic.hydro_generator import (
    generate_network, generate_corpus,
    _dirty_length, _dirty_diameter, _fitting_loss_factor,
    DN_SCHEDULE, FITTING_LOSS_FACTORS, TOPOLOGY_VARIANTS,
)


class TestDirtyValues:
    def test_dirty_length_never_below_minimum(self, rng):
        """All generated lengths >= 0.05 m."""
        for _ in range(1000):
            assert _dirty_length(1.0, rng) >= 0.05

    def test_dirty_length_distribution_not_all_round(self, rng):
        """Fewer than 5% of 1000 samples equal the exact base value."""
        base = 2.0
        lengths = [_dirty_length(base, rng) for _ in range(1000)]
        exact_matches = sum(1 for l in lengths if l == base)
        assert exact_matches < 50  # < 5%

    def test_dirty_diameter_75pct_clean(self, rng):
        """Between 70% and 80% of 1000 samples are exact nominal values."""
        nominal = 25.0
        diameters = [_dirty_diameter(nominal, rng) for _ in range(1000)]
        clean = sum(1 for d in diameters if d == nominal)
        assert 700 <= clean <= 800

    def test_dirty_diameter_always_positive(self, rng):
        """All generated diameters > 0."""
        for _ in range(1000):
            assert _dirty_diameter(15.0, rng) > 0

    def test_fitting_loss_factor_within_range(self, rng):
        """Loss factors are within 85%-115% of base values."""
        for ft, base in FITTING_LOSS_FACTORS.items():
            for _ in range(100):
                val = _fitting_loss_factor(ft, rng)
                assert base * 0.84 <= val <= base * 1.16  # small float margin


class TestGenerateNetwork:
    def test_is_tree(self):
        """Generated graph is a tree."""
        rng = np.random.default_rng(42)
        G = generate_network(
            rng, n_floors=2, fixtures_per_floor=[2, 1],
            tank_elevation_m=12.0,
            nominal_diameters_mm={"trunk": 40, "riser": 32, "branch": 25, "stub": 20},
        )
        assert nx.is_tree(G.to_undirected())

    def test_exactly_one_tank(self):
        """Exactly one IfcTank node."""
        rng = np.random.default_rng(42)
        G = generate_network(
            rng, n_floors=1, fixtures_per_floor=[3],
            tank_elevation_m=10.0,
            nominal_diameters_mm={"trunk": 40, "riser": 32, "branch": 25, "stub": 20},
        )
        tanks = [n for n, d in G.nodes(data=True) if d["ifc_type"] == "IfcTank"]
        assert len(tanks) == 1

    def test_terminal_count_matches(self):
        """Number of terminals equals sum of fixtures_per_floor."""
        rng = np.random.default_rng(42)
        fixtures = [3, 2]
        G = generate_network(
            rng, n_floors=2, fixtures_per_floor=fixtures,
            tank_elevation_m=12.0,
            nominal_diameters_mm={"trunk": 40, "riser": 32, "branch": 25, "stub": 20},
        )
        terms = [n for n, d in G.nodes(data=True) if d["ifc_type"] == "IfcSanitaryTerminal"]
        assert len(terms) == sum(fixtures)

    def test_all_nodes_have_required_attributes(self):
        """Every node has global_id, ifc_type, x, y, z, elevation_m."""
        rng = np.random.default_rng(42)
        G = generate_network(
            rng, n_floors=2, fixtures_per_floor=[2, 2],
            tank_elevation_m=12.0,
            nominal_diameters_mm={"trunk": 40, "riser": 32, "branch": 25, "stub": 20},
        )
        required = {"global_id", "ifc_type", "x", "y", "z", "elevation_m"}
        for node, data in G.nodes(data=True):
            assert required.issubset(data.keys()), f"Node {node} missing: {required - data.keys()}"

    def test_pipe_nodes_have_length_and_diameter(self):
        """Every IfcPipeSegment has length_m > 0 and nominal_diameter_mm > 0."""
        rng = np.random.default_rng(42)
        G = generate_network(
            rng, n_floors=2, fixtures_per_floor=[2, 1],
            tank_elevation_m=12.0,
            nominal_diameters_mm={"trunk": 40, "riser": 32, "branch": 25, "stub": 20},
        )
        for n, d in G.nodes(data=True):
            if d["ifc_type"] == "IfcPipeSegment":
                assert d["length_m"] is not None and d["length_m"] > 0
                assert d["nominal_diameter_mm"] is not None and d["nominal_diameter_mm"] > 0

    def test_fitting_nodes_have_loss_factor(self):
        """Every IfcPipeFitting has fitting_loss_factor > 0."""
        rng = np.random.default_rng(42)
        G = generate_network(
            rng, n_floors=2, fixtures_per_floor=[2, 1],
            tank_elevation_m=12.0,
            nominal_diameters_mm={"trunk": 40, "riser": 32, "branch": 25, "stub": 20},
        )
        for n, d in G.nodes(data=True):
            if d["ifc_type"] == "IfcPipeFitting":
                assert d["fitting_loss_factor"] is not None and d["fitting_loss_factor"] > 0

    def test_invalid_fixtures_per_floor_raises(self):
        """ValueError if len(fixtures_per_floor) != n_floors."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError):
            generate_network(
                rng, n_floors=2, fixtures_per_floor=[1],
                tank_elevation_m=10.0,
                nominal_diameters_mm={"trunk": 40, "riser": 32, "branch": 25, "stub": 20},
            )


class TestGenerateCorpus:
    def test_corpus_size(self):
        """generate_corpus(n=20) returns 20 graphs."""
        corpus = generate_corpus(n=20, seed=42)
        assert len(corpus) == 20

    def test_corpus_has_labels(self):
        """Each entry has system_conforms and path_labels."""
        corpus = generate_corpus(n=10, seed=42)
        for graph, labels in corpus:
            assert "system_conforms" in labels
            assert "path_labels" in labels

    def test_corpus_reproducible(self):
        """Two calls with same seed produce identical results."""
        c1 = generate_corpus(n=10, seed=42)
        c2 = generate_corpus(n=10, seed=42)
        for (g1, l1), (g2, l2) in zip(c1, c2):
            assert l1["model_id"] == l2["model_id"]
            assert l1["system_conforms"] == l2["system_conforms"]

    def test_corpus_nonconforming_fraction(self):
        """Nonconforming fraction is between 0.20 and 0.60 for n=50."""
        corpus = generate_corpus(n=50, seed=42, target_nonconforming_fraction=0.40)
        nc = sum(1 for _, l in corpus if not l["system_conforms"])
        frac = nc / len(corpus)
        assert 0.20 <= frac <= 0.60, f"Nonconforming fraction {frac} out of range"

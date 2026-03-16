"""Shared pytest fixtures for ifc-hydro-topologic tests."""
from __future__ import annotations

import uuid
import numpy as np
import networkx as nx
import pytest


@pytest.fixture
def simple_3node_graph() -> nx.DiGraph:
    """Minimal valid hydraulic graph: Tank -> Pipe -> Terminal."""
    G = nx.DiGraph()
    tank_id = str(uuid.uuid4())
    pipe_id = str(uuid.uuid4())
    term_id = str(uuid.uuid4())

    G.add_node(tank_id, ifc_type="IfcTank", global_id=tank_id,
               x=0.0, y=0.0, z=10.0, elevation_m=10.0,
               nominal_diameter_mm=None, length_m=None,
               fitting_type=None, fitting_loss_factor=None, fixture_type=None)
    G.add_node(pipe_id, ifc_type="IfcPipeSegment", global_id=pipe_id,
               x=0.0, y=0.0, z=5.0, elevation_m=5.0,
               nominal_diameter_mm=25.0, length_m=3.5,
               fitting_type=None, fitting_loss_factor=None, fixture_type=None)
    G.add_node(term_id, ifc_type="IfcSanitaryTerminal", global_id=term_id,
               x=1.0, y=0.0, z=1.0, elevation_m=1.0,
               nominal_diameter_mm=None, length_m=None,
               fitting_type=None, fitting_loss_factor=None, fixture_type="washbasin")

    G.add_edge(tank_id, pipe_id, length_m=3.5, nominal_diameter_mm=25.0, fitting_loss_factor=0.0)
    G.add_edge(pipe_id, term_id, length_m=0.0, nominal_diameter_mm=25.0, fitting_loss_factor=0.0)
    return G


@pytest.fixture
def synthetic_tree_graph() -> nx.DiGraph:
    """
    A 2-floor, 2-fixtures-per-floor tree graph (~15 nodes).
    Tank at z=12m, floor 1 at z=3m, floor 2 at z=6m.
    Uses realistic dirty values.
    """
    G = nx.DiGraph()
    rng = np.random.default_rng(42)

    def _add(ifc_type, z, **kw):
        gid = str(uuid.uuid4())
        G.add_node(gid, ifc_type=ifc_type, global_id=gid,
                   x=kw.pop("x", rng.uniform(0, 5)),
                   y=kw.pop("y", rng.uniform(0, 5)),
                   z=z, elevation_m=z, **kw)
        return gid

    tank = _add("IfcTank", 12.0, nominal_diameter_mm=None, length_m=None,
                fitting_type=None, fitting_loss_factor=None, fixture_type=None)

    # Trunk pipe
    trunk = _add("IfcPipeSegment", 10.0, nominal_diameter_mm=40.0, length_m=2.347,
                 fitting_type=None, fitting_loss_factor=None, fixture_type=None)
    G.add_edge(tank, trunk, length_m=2.347, nominal_diameter_mm=40.0, fitting_loss_factor=0.0)

    # Floor 1 tee
    tee1 = _add("IfcPipeFitting", 6.0, nominal_diameter_mm=40.0, length_m=None,
                fitting_type="tee_branch", fitting_loss_factor=1.42, fixture_type=None)
    G.add_edge(trunk, tee1, length_m=0.0, nominal_diameter_mm=40.0, fitting_loss_factor=1.42)

    # Floor 1 branches
    for i in range(2):
        branch = _add("IfcPipeSegment", 3.0, nominal_diameter_mm=25.0, length_m=1.5 + i * 0.8,
                       fitting_type=None, fitting_loss_factor=None, fixture_type=None)
        G.add_edge(tee1, branch, length_m=1.5 + i * 0.8, nominal_diameter_mm=25.0,
                   fitting_loss_factor=0.0)
        term = _add("IfcSanitaryTerminal", 1.0, nominal_diameter_mm=None, length_m=None,
                     fitting_type=None, fitting_loss_factor=None,
                     fixture_type=["shower", "washbasin"][i])
        G.add_edge(branch, term, length_m=0.0, nominal_diameter_mm=25.0, fitting_loss_factor=0.0)

    # Riser to floor 2
    riser = _add("IfcPipeSegment", 6.0, nominal_diameter_mm=32.0, length_m=3.0,
                 fitting_type=None, fitting_loss_factor=None, fixture_type=None)
    G.add_edge(tee1, riser, length_m=3.0, nominal_diameter_mm=32.0, fitting_loss_factor=0.0)

    tee2 = _add("IfcPipeFitting", 9.0, nominal_diameter_mm=32.0, length_m=None,
                fitting_type="tee_branch", fitting_loss_factor=1.55, fixture_type=None)
    G.add_edge(riser, tee2, length_m=0.0, nominal_diameter_mm=32.0, fitting_loss_factor=1.55)

    for i in range(2):
        branch = _add("IfcPipeSegment", 6.0, nominal_diameter_mm=20.0, length_m=1.2 + i * 0.5,
                       fitting_type=None, fitting_loss_factor=None, fixture_type=None)
        G.add_edge(tee2, branch, length_m=1.2 + i * 0.5, nominal_diameter_mm=20.0,
                   fitting_loss_factor=0.0)
        term = _add("IfcSanitaryTerminal", 6.0, nominal_diameter_mm=None, length_m=None,
                     fitting_type=None, fitting_loss_factor=None,
                     fixture_type=["wc", "kitchen_sink"][i])
        G.add_edge(branch, term, length_m=0.0, nominal_diameter_mm=20.0, fitting_loss_factor=0.0)

    return G


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded RNG for deterministic tests."""
    return np.random.default_rng(12345)

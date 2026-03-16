"""Synthetic hydraulic network generator for training data production."""
from __future__ import annotations

import logging
import math
import uuid
from typing import Any

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# --- Constants ---

DN_SCHEDULE = [15, 20, 25, 32, 40, 50, 65, 80]

FITTING_LOSS_FACTORS: dict[str, float] = {
    "elbow_90": 1.5,
    "elbow_45": 0.4,
    "tee_branch": 1.8,
    "tee_run": 0.5,
    "reducer": 0.25,
    "valve_gate": 0.2,
}

FIXTURE_TYPES = ["washbasin", "shower", "wc", "kitchen_sink", "bidet", "laundry"]

# Design flow scores per terminal type (from ifc-hydro NBR 5626 tables).
# Keys match fixture_type values used in generator.
SCORE_TABLE: dict[str, float] = {
    "washbasin": 0.3,
    "shower": 0.1,
    "wc": 0.3,
    "kitchen_sink": 0.7,
    "bidet": 0.1,
    "laundry": 1.0,
}

# Internal diameter table (nominal mm -> internal mm) from ifc-hydro
INTERNAL_DIAMETER_TABLE: dict[float, float] = {
    15: 17.0,
    20: 21.6,
    25: 27.8,
    32: 35.2,
    40: 44.0,
    50: 53.4,
    65: 66.6,
    75: 75.6,
    80: 75.6,  # Approximate for DN80
}

# Fitting equivalent length table (fitting_type -> {nominal_diameter_mm -> equiv_length_m})
FITTING_EQUIV_LENGTH: dict[str, dict[float, float]] = {
    "tee_branch": {15: 2.3, 20: 2.4, 25: 3.1, 32: 3.5, 40: 4.6, 50: 5.1, 65: 6.9, 80: 9.5},
    "tee_run": {15: 0.7, 20: 0.8, 25: 0.9, 32: 1.5, 40: 1.5, 50: 2.2, 65: 2.3, 80: 3.4},
    "elbow_90": {15: 1.1, 20: 1.2, 25: 1.5, 32: 2.0, 40: 2.1, 50: 2.7, 65: 3.4, 80: 4.9},
    "elbow_45": {15: 0.4, 20: 0.5, 25: 0.7, 32: 0.9, 40: 1.0, 50: 1.3, 65: 1.6, 80: 2.4},
    "reducer": {15: 0.2, 20: 0.3, 25: 0.4, 32: 0.5, 40: 0.5, 50: 0.7, 65: 0.8, 80: 1.3},
    "valve_gate": {15: 0.1, 20: 0.1, 25: 0.2, 32: 0.2, 40: 0.3, 50: 0.3, 65: 0.4, 80: 0.6},
}

TOPOLOGY_VARIANTS = [
    {"n_floors": 1, "fixtures_per_floor": [1]},
    {"n_floors": 1, "fixtures_per_floor": [2]},
    {"n_floors": 1, "fixtures_per_floor": [3]},
    {"n_floors": 2, "fixtures_per_floor": [2, 1]},
    {"n_floors": 2, "fixtures_per_floor": [2, 2]},
    {"n_floors": 2, "fixtures_per_floor": [3, 2]},
    {"n_floors": 3, "fixtures_per_floor": [2, 2, 1]},
    {"n_floors": 3, "fixtures_per_floor": [3, 2, 2]},
    {"n_floors": 3, "fixtures_per_floor": [3, 3, 3]},
    {"n_floors": 4, "fixtures_per_floor": [3, 3, 2, 2]},
]


# --- Dirty value generators ---

def _dirty_length(base_length_m: float, rng: np.random.Generator) -> float:
    """
    Apply lognormal perturbation to a base pipe length.
    Returns a realistic 'measured' pipe length >= 0.05 m.
    """
    log_noise = rng.normal(0.0, 0.1)
    dirty = base_length_m * math.exp(log_noise)
    return max(dirty, 0.05)


def _dirty_diameter(nominal_mm: float, rng: np.random.Generator) -> float:
    """
    Return nominal diameter 75% of the time, or perturbed diameter 25%.
    Always returns a positive value.
    """
    if rng.random() < 0.75:
        return nominal_mm
    # Perturb by ±10%
    factor = rng.uniform(0.90, 1.10)
    return max(nominal_mm * factor, 1.0)


def _fitting_loss_factor(fitting_type: str, rng: np.random.Generator) -> float:
    """
    Return a fitting loss factor with ±15% uniform perturbation from the base value.
    """
    base = FITTING_LOSS_FACTORS.get(fitting_type, 1.0)
    factor = rng.uniform(0.85, 1.15)
    return base * factor


# --- Network generator ---

def generate_network(
    rng: np.random.Generator,
    n_floors: int,
    fixtures_per_floor: list[int],
    tank_elevation_m: float,
    nominal_diameters_mm: dict[str, float],
    floor_height_m: float = 3.0,
) -> nx.DiGraph:
    """
    Generate a single synthetic plumbing network as a rooted tree DiGraph.

    Args:
        rng: Seeded random number generator.
        n_floors: Number of floors in the building.
        fixtures_per_floor: List of fixture counts per floor. len must == n_floors.
        tank_elevation_m: Elevation of the tank in metres.
        nominal_diameters_mm: Dict with keys 'trunk', 'riser', 'branch', 'stub'.
        floor_height_m: Height per floor in metres. Default 3.0.

    Returns:
        nx.DiGraph representing the plumbing tree.

    Raises:
        ValueError: If len(fixtures_per_floor) != n_floors.
    """
    if len(fixtures_per_floor) != n_floors:
        raise ValueError(
            f"len(fixtures_per_floor)={len(fixtures_per_floor)} != n_floors={n_floors}"
        )

    G = nx.DiGraph()
    x_cursor = 0.0

    def _add_node(ifc_type: str, x: float, y: float, z: float, **kw) -> str:
        gid = str(uuid.uuid4())
        # Add tiny jitter to avoid coordinate collisions
        x += rng.uniform(-1e-4, 1e-4)
        y += rng.uniform(-1e-4, 1e-4)
        G.add_node(
            gid,
            ifc_type=ifc_type,
            global_id=gid,
            x=x, y=y, z=z,
            elevation_m=z,
            **kw,
        )
        return gid

    def _add_edge(src: str, tgt: str) -> None:
        src_data = G.nodes[src]
        tgt_data = G.nodes[tgt]

        # Edge attribute propagation rules
        if src_data["ifc_type"] == "IfcPipeSegment":
            length = src_data.get("length_m", 0.0) or 0.0
        else:
            length = 0.0

        diam = src_data.get("nominal_diameter_mm") or tgt_data.get("nominal_diameter_mm")

        if src_data["ifc_type"] == "IfcPipeFitting":
            loss = src_data.get("fitting_loss_factor", 0.0) or 0.0
        else:
            loss = 0.0

        G.add_edge(src, tgt,
                    length_m=length,
                    nominal_diameter_mm=diam,
                    fitting_loss_factor=loss)

    # Tank at top
    tank_id = _add_node(
        "IfcTank", x=0.0, y=0.0, z=tank_elevation_m,
        nominal_diameter_mm=None, length_m=None,
        fitting_type=None, fitting_loss_factor=None, fixture_type=None,
    )

    # Trunk pipe from tank down
    trunk_length = _dirty_length(tank_elevation_m - (n_floors * floor_height_m), rng)
    trunk_z = tank_elevation_m - trunk_length
    trunk_diam = _dirty_diameter(nominal_diameters_mm["trunk"], rng)
    trunk_id = _add_node(
        "IfcPipeSegment", x=0.0, y=0.0, z=trunk_z,
        nominal_diameter_mm=trunk_diam, length_m=trunk_length,
        fitting_type=None, fitting_loss_factor=None, fixture_type=None,
    )
    _add_edge(tank_id, trunk_id)

    # Build floor by floor from top floor down
    prev_node = trunk_id
    for floor_idx in range(n_floors):
        floor_num = n_floors - floor_idx  # top floor first
        floor_z = floor_num * floor_height_m
        n_fixtures = fixtures_per_floor[floor_idx]
        x_cursor = 0.0

        # Tee fitting at floor junction
        tee_diam = _dirty_diameter(nominal_diameters_mm.get("riser", 32), rng)
        tee_loss = _fitting_loss_factor("tee_branch", rng)
        tee_id = _add_node(
            "IfcPipeFitting", x=x_cursor, y=0.0, z=floor_z,
            nominal_diameter_mm=tee_diam, length_m=None,
            fitting_type="tee_branch", fitting_loss_factor=tee_loss, fixture_type=None,
        )
        _add_edge(prev_node, tee_id)

        # Branches to fixtures on this floor
        for fix_idx in range(n_fixtures):
            x_offset = (fix_idx + 1) * 1.5
            branch_length = _dirty_length(1.0 + fix_idx * 0.5, rng)
            branch_diam = _dirty_diameter(nominal_diameters_mm["branch"], rng)

            branch_id = _add_node(
                "IfcPipeSegment", x=x_offset, y=rng.uniform(0, 2), z=floor_z - 0.3,
                nominal_diameter_mm=branch_diam, length_m=branch_length,
                fitting_type=None, fitting_loss_factor=None, fixture_type=None,
            )
            _add_edge(tee_id, branch_id)

            # Stub to terminal
            stub_length = _dirty_length(0.5, rng)
            stub_diam = _dirty_diameter(nominal_diameters_mm["stub"], rng)
            stub_id = _add_node(
                "IfcPipeSegment", x=x_offset + 0.5, y=rng.uniform(0, 2), z=floor_z - 0.5,
                nominal_diameter_mm=stub_diam, length_m=stub_length,
                fitting_type=None, fitting_loss_factor=None, fixture_type=None,
            )
            _add_edge(branch_id, stub_id)

            # Terminal fixture
            fixture_type = FIXTURE_TYPES[rng.integers(0, len(FIXTURE_TYPES))]
            term_id = _add_node(
                "IfcSanitaryTerminal",
                x=x_offset + 1.0, y=rng.uniform(0, 2), z=floor_z - 0.8,
                nominal_diameter_mm=None, length_m=None,
                fitting_type=None, fitting_loss_factor=None, fixture_type=fixture_type,
            )
            _add_edge(stub_id, term_id)

        # Riser to next floor (if not last floor)
        if floor_idx < n_floors - 1:
            riser_length = _dirty_length(floor_height_m, rng)
            riser_diam = _dirty_diameter(nominal_diameters_mm.get("riser", 32), rng)
            riser_id = _add_node(
                "IfcPipeSegment",
                x=0.0, y=0.0, z=floor_z - riser_length,
                nominal_diameter_mm=riser_diam, length_m=riser_length,
                fitting_type=None, fitting_loss_factor=None, fixture_type=None,
            )
            _add_edge(tee_id, riser_id)
            prev_node = riser_id

    return G


# --- Conformance checker ---

def _simple_conformance_check(graph: nx.DiGraph) -> dict:
    """
    Simplified conformance check when HydraulicCalculator is unavailable.

    For each path from IfcTank to IfcSanitaryTerminal:
    1. Available pressure = rho * g * (tank_elevation - terminal_elevation)
       = 9.81 * (z_tank - z_terminal) kPa  (assuming rho=1000 kg/m^3)
    2. Pressure loss = sum of friction losses along path
       Using Fair-Whipple-Hsiao: J = 0.000869 * Q^1.75 * D_int^-4.75
       where Q = 0.3 * sqrt(score_sum) in L/s, D_int in metres
    3. System conforms if available pressure > total loss + 10 kPa minimum

    This is a FALLBACK. The real HydraulicCalculator uses proper NBR 5626 tables.

    Returns:
        dict with keys:
            'system_conforms': bool
            'path_labels': dict mapping terminal_global_id -> bool
            'terminal_pressures': dict mapping terminal_global_id -> float (remaining pressure in mWC)
    """
    # Find tank and terminals
    tank_id = None
    terminal_ids = []
    for node, data in graph.nodes(data=True):
        if data.get("ifc_type") == "IfcTank":
            tank_id = node
        elif data.get("ifc_type") == "IfcSanitaryTerminal":
            terminal_ids.append(node)

    if tank_id is None:
        raise ValueError("No IfcTank found in graph.")
    if not terminal_ids:
        raise ValueError("No IfcSanitaryTerminal found in graph.")

    tank_z = graph.nodes[tank_id].get("z", graph.nodes[tank_id].get("elevation_m", 0.0))
    # Tank height adjustment (water level above outlet)
    tank_height_adj = 0.5

    path_labels = {}
    terminal_pressures = {}

    for term_id in terminal_ids:
        try:
            path = nx.shortest_path(graph, tank_id, term_id)
        except nx.NetworkXNoPath:
            path_labels[graph.nodes[term_id]["global_id"]] = False
            terminal_pressures[graph.nodes[term_id]["global_id"]] = 0.0
            continue

        term_data = graph.nodes[term_id]
        term_z = term_data.get("z", term_data.get("elevation_m", 0.0))
        term_gid = term_data["global_id"]

        # Available pressure head (metres of water column)
        available = (tank_z + tank_height_adj) - term_z

        # Get fixture score for design flow calculation
        fixture_type = term_data.get("fixture_type", "washbasin")
        score = SCORE_TABLE.get(fixture_type, 0.3)

        # Walk path and sum pressure losses
        total_loss = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            u_data = graph.nodes[u]
            edge_data = graph.edges[u, v]

            ifc_type = u_data.get("ifc_type", "")

            if ifc_type == "IfcPipeSegment":
                length = edge_data.get("length_m", 0.0) or 0.0
                diam_mm = u_data.get("nominal_diameter_mm")
                if diam_mm and diam_mm > 0:
                    # Map to internal diameter
                    internal_mm = INTERNAL_DIAMETER_TABLE.get(
                        round(diam_mm), diam_mm * 0.88
                    )
                    internal_m = internal_mm / 1000.0

                    # Fair-Whipple-Hsiao
                    design_flow = 0.3 * (score ** 0.5)  # L/s
                    if internal_m > 0 and design_flow > 0:
                        unit_loss = 0.000869 * ((design_flow * 0.001) ** 1.75) * (
                            internal_m ** -4.75
                        )
                        total_loss += length * unit_loss

            elif ifc_type == "IfcPipeFitting":
                fitting_type = u_data.get("fitting_type", "tee_branch")
                diam_mm = u_data.get("nominal_diameter_mm")
                if diam_mm and diam_mm > 0:
                    internal_mm = INTERNAL_DIAMETER_TABLE.get(
                        round(diam_mm), diam_mm * 0.88
                    )
                    internal_m = internal_mm / 1000.0
                    equiv_lengths = FITTING_EQUIV_LENGTH.get(fitting_type, {})
                    equiv = equiv_lengths.get(round(diam_mm), 1.0)

                    design_flow = 0.3 * (score ** 0.5)
                    if internal_m > 0 and design_flow > 0:
                        unit_loss = 0.000869 * ((design_flow * 0.001) ** 1.75) * (
                            internal_m ** -4.75
                        )
                        total_loss += equiv * unit_loss

        remaining = available - total_loss
        # Conform if remaining >= 1.0 mWC (~10 kPa)
        conforms = remaining >= 1.0
        path_labels[term_gid] = conforms
        terminal_pressures[term_gid] = remaining

    system_conforms = all(path_labels.values())

    return {
        "system_conforms": system_conforms,
        "path_labels": path_labels,
        "terminal_pressures": terminal_pressures,
    }


# --- Corpus generator ---

def generate_corpus(
    n: int,
    seed: int,
    target_nonconforming_fraction: float = 0.40,
    output_dir: str | None = None,
) -> list[tuple[nx.DiGraph, dict]]:
    """
    Generate a corpus of synthetic plumbing networks with conformance labels.

    Args:
        n: Number of graphs to generate.
        seed: Master random seed for reproducibility.
        target_nonconforming_fraction: Target fraction of non-conforming systems (0.30-0.50).
        output_dir: Optional directory to save individual graphs (not used currently).

    Returns:
        List of (graph, labels) tuples where labels is a dict with:
            'model_id': str (uuid)
            'system_conforms': bool
            'path_labels': dict[str, bool]
            'terminal_pressures': dict[str, float]
    """
    master_rng = np.random.default_rng(seed)
    corpus: list[tuple[nx.DiGraph, dict]] = []

    nc_count = 0

    for i in range(n):
        # Adaptive biasing
        current_frac = nc_count / max(i, 1)

        # Choose topology variant
        variant_idx = master_rng.integers(0, len(TOPOLOGY_VARIANTS))
        variant = TOPOLOGY_VARIANTS[variant_idx]

        # Bias diameters and tank elevation based on current conformance fraction
        if current_frac < target_nonconforming_fraction * 0.8:
            # Need more non-conforming: use smaller pipes, lower tank
            tank_elev = master_rng.uniform(5.0, 8.0)
            diameters = {
                "trunk": master_rng.choice([20, 25]),
                "riser": master_rng.choice([15, 20]),
                "branch": master_rng.choice([15, 20]),
                "stub": 15,
            }
        elif current_frac > target_nonconforming_fraction * 1.2:
            # Need more conforming: use larger pipes, higher tank
            tank_elev = master_rng.uniform(15.0, 25.0)
            diameters = {
                "trunk": master_rng.choice([50, 65]),
                "riser": master_rng.choice([40, 50]),
                "branch": master_rng.choice([32, 40]),
                "stub": master_rng.choice([25, 32]),
            }
        else:
            # Normal range
            tank_elev = master_rng.uniform(8.0, 18.0)
            diameters = {
                "trunk": master_rng.choice(DN_SCHEDULE[3:]),  # 32+
                "riser": master_rng.choice(DN_SCHEDULE[2:5]),  # 25-40
                "branch": master_rng.choice(DN_SCHEDULE[1:4]),  # 20-32
                "stub": master_rng.choice(DN_SCHEDULE[:3]),  # 15-25
            }

        graph_rng = np.random.default_rng(master_rng.integers(0, 2**31))

        G = generate_network(
            rng=graph_rng,
            n_floors=variant["n_floors"],
            fixtures_per_floor=variant["fixtures_per_floor"],
            tank_elevation_m=tank_elev,
            nominal_diameters_mm={k: float(v) for k, v in diameters.items()},
        )

        # Run conformance check
        check = _simple_conformance_check(G)

        # Generate deterministic model_id from the master RNG
        uid_bytes = bytes(master_rng.integers(0, 256, size=16, dtype=np.uint8))
        model_id = str(uuid.UUID(bytes=uid_bytes))
        labels = {
            "model_id": model_id,
            "system_conforms": check["system_conforms"],
            "path_labels": check["path_labels"],
            "terminal_pressures": check["terminal_pressures"],
        }

        corpus.append((G, labels))

        if not check["system_conforms"]:
            nc_count += 1

    nc_frac = nc_count / n if n > 0 else 0.0
    logger.info(
        "Generated %d graphs: %d conforming, %d non-conforming (%.1f%% NC).",
        n, n - nc_count, nc_count, nc_frac * 100,
    )

    return corpus

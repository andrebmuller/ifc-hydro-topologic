"""Serialize graph corpus to CSV format for topologicpy PyG training."""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# IFC type to one-hot index mapping
IFC_TYPE_MAP: dict[str, int] = {
    "IfcTank": 0,
    "IfcPipeSegment": 1,
    "IfcPipeFitting": 2,
    "IfcSanitaryTerminal": 3,
    "IfcValve": 4,
}
NUM_IFC_TYPES = 5


def _compute_depth_from_root(G: nx.DiGraph) -> dict[str, int]:
    """Compute BFS depth from the tank (root) node for each node."""
    root = None
    for node, data in G.nodes(data=True):
        if data.get("ifc_type") == "IfcTank":
            root = node
            break
    if root is None:
        return {data["global_id"]: 0 for _, data in G.nodes(data=True)}

    depths = {}
    undirected = G.to_undirected()
    for node, depth in nx.shortest_path_length(undirected, root).items():
        gid = G.nodes[node].get("global_id", str(node))
        depths[gid] = depth
    return depths


def serialize_dataset_to_pyg_csv(
    corpus: list[tuple[nx.DiGraph, dict]],
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict:
    """
    Serialize a graph corpus to CSV files for PyG training.

    Produces three files:
    - graphs.csv: graph_id, label, feat_0..feat_k
    - nodes.csv: graph_id, node_id, feat_*, label, train_mask, val_mask, test_mask
    - edges.csv: graph_id, src_id, dst_id, feat_*

    Feature engineering:
    - One-hot encode ifc_type (5 columns)
    - Normalize nominal_diameter_mm to [0, 1]
    - Normalize length_m to [0, 1] using 99th percentile
    - Z-score normalize elevation_m
    - depth_from_root (integer)
    - is_terminal (binary)
    - betweenness_centrality

    Args:
        corpus: List of (graph, labels) tuples from generate_corpus().
        output_dir: Directory to write CSV files.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.

    Returns:
        dict with keys: 'graphs_csv', 'nodes_csv', 'edges_csv', 'class_balance',
                        'n_features', 'n_graphs'
    """
    os.makedirs(output_dir, exist_ok=True)

    # First pass: collect statistics for normalization
    all_diameters = []
    all_lengths = []
    all_elevations = []

    for G, _ in corpus:
        for _, data in G.nodes(data=True):
            diam = data.get("nominal_diameter_mm")
            if diam is not None and diam > 0:
                all_diameters.append(diam)
            length = data.get("length_m")
            if length is not None and length > 0:
                all_lengths.append(length)
            elev = data.get("elevation_m", data.get("z", 0.0))
            if elev is not None:
                all_elevations.append(elev)

    max_diameter = max(all_diameters) if all_diameters else 1.0
    length_99 = float(np.percentile(all_lengths, 99)) if all_lengths else 1.0
    elev_mean = float(np.mean(all_elevations)) if all_elevations else 0.0
    elev_std = float(np.std(all_elevations)) if all_elevations else 1.0
    if elev_std < 1e-6:
        elev_std = 1.0

    # Stratified train/val/test split at graph level
    n = len(corpus)
    labels_arr = np.array([1 if lab["system_conforms"] else 0 for _, lab in corpus])

    # Deterministic stratified split
    rng = np.random.default_rng(42)
    indices = np.arange(n)
    rng.shuffle(indices)

    # Sort by label to stratify
    pos_idx = indices[labels_arr[indices] == 1]
    neg_idx = indices[labels_arr[indices] == 0]

    def _split_indices(idx_arr):
        n_i = len(idx_arr)
        n_train = int(n_i * train_ratio)
        n_val = int(n_i * val_ratio)
        return (
            idx_arr[:n_train],
            idx_arr[n_train:n_train + n_val],
            idx_arr[n_train + n_val:],
        )

    pos_train, pos_val, pos_test = _split_indices(pos_idx)
    neg_train, neg_val, neg_test = _split_indices(neg_idx)

    train_set = set(np.concatenate([pos_train, neg_train]))
    val_set = set(np.concatenate([pos_val, neg_val]))
    test_set = set(np.concatenate([pos_test, neg_test]))

    # Build dataframes
    graph_rows = []
    node_rows = []
    edge_rows = []

    for graph_idx, (G, labels) in enumerate(corpus):
        system_label = 1 if labels["system_conforms"] else 0
        path_labels = labels.get("path_labels", {})

        # Determine split
        if graph_idx in train_set:
            split = "train"
        elif graph_idx in val_set:
            split = "val"
        else:
            split = "test"

        # Graph-level row
        graph_rows.append({
            "graph_id": graph_idx,
            "label": system_label,
            "split": split,
        })

        # Compute per-graph metrics
        depths = _compute_depth_from_root(G)
        bc = nx.betweenness_centrality(G, normalized=True)

        # Build node ID mapping for edge indices
        node_id_map = {}

        for node_local_idx, (node, data) in enumerate(G.nodes(data=True)):
            gid = data.get("global_id", str(node))
            node_id_map[node] = node_local_idx

            # Features
            ifc_type = data.get("ifc_type", "Unknown")
            type_idx = IFC_TYPE_MAP.get(ifc_type, -1)
            one_hot = [0] * NUM_IFC_TYPES
            if 0 <= type_idx < NUM_IFC_TYPES:
                one_hot[type_idx] = 1

            diam = data.get("nominal_diameter_mm")
            feat_diam = (diam / max_diameter) if diam is not None and diam > 0 else 0.0

            length = data.get("length_m")
            feat_length = (length / length_99) if length is not None and length > 0 else 0.0
            feat_length = min(feat_length, 1.0)

            elev = data.get("elevation_m", data.get("z", 0.0))
            feat_elev = (elev - elev_mean) / elev_std if elev is not None else 0.0

            feat_depth = depths.get(gid, 0)
            feat_is_terminal = 1 if ifc_type == "IfcSanitaryTerminal" else 0
            feat_bc = bc.get(node, 0.0)

            # Node label: -1 for non-terminals, 0/1 for terminals
            if ifc_type == "IfcSanitaryTerminal":
                node_label = 1 if path_labels.get(gid, True) else 0
            else:
                node_label = -1

            # Masks
            train_mask = split == "train" and feat_is_terminal == 1
            val_mask = split == "val" and feat_is_terminal == 1
            test_mask = split == "test" and feat_is_terminal == 1

            row = {
                "graph_id": graph_idx,
                "node_id": node_local_idx,
                "feat_ifc_type_0": one_hot[0],
                "feat_ifc_type_1": one_hot[1],
                "feat_ifc_type_2": one_hot[2],
                "feat_ifc_type_3": one_hot[3],
                "feat_ifc_type_4": one_hot[4],
                "feat_nominal_diameter_mm": feat_diam,
                "feat_length_m": feat_length,
                "feat_elevation_m": feat_elev,
                "feat_depth_from_root": feat_depth,
                "feat_is_terminal": feat_is_terminal,
                "feat_betweenness_centrality": feat_bc,
                "label": node_label,
                "train_mask": train_mask,
                "val_mask": val_mask,
                "test_mask": test_mask,
            }
            node_rows.append(row)

        # Edges
        for u, v, data in G.edges(data=True):
            src_local = node_id_map.get(u, -1)
            dst_local = node_id_map.get(v, -1)

            diam = data.get("nominal_diameter_mm")
            feat_diam = (diam / max_diameter) if diam is not None and diam > 0 else 0.0

            length = data.get("length_m", 0.0) or 0.0
            feat_length = (length / length_99) if length > 0 else 0.0
            feat_length = min(feat_length, 1.0)

            loss = data.get("fitting_loss_factor", 0.0) or 0.0

            edge_rows.append({
                "graph_id": graph_idx,
                "src_id": src_local,
                "dst_id": dst_local,
                "feat_length_m": feat_length,
                "feat_nominal_diameter_mm": feat_diam,
                "feat_fitting_loss_factor": loss,
            })

    # Write CSVs
    graphs_df = pd.DataFrame(graph_rows)
    nodes_df = pd.DataFrame(node_rows)
    edges_df = pd.DataFrame(edge_rows)

    graphs_path = os.path.join(output_dir, "graphs.csv")
    nodes_path = os.path.join(output_dir, "nodes.csv")
    edges_path = os.path.join(output_dir, "edges.csv")

    graphs_df.to_csv(graphs_path, index=False)
    nodes_df.to_csv(nodes_path, index=False)
    edges_df.to_csv(edges_path, index=False)

    # Class balance
    class_balance = {
        0: int((labels_arr == 0).sum()),
        1: int((labels_arr == 1).sum()),
    }

    feat_cols = [c for c in nodes_df.columns if c.startswith("feat_")]

    logger.info(
        "Serialized %d graphs to %s (%d nodes, %d edges, %d features).",
        len(corpus), output_dir, len(nodes_df), len(edges_df), len(feat_cols),
    )

    return {
        "graphs_csv": graphs_path,
        "nodes_csv": nodes_path,
        "edges_csv": edges_path,
        "class_balance": class_balance,
        "n_features": len(feat_cols),
        "n_graphs": len(corpus),
    }

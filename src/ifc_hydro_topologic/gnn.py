"""GNN training wrapper for MEP conformance prediction."""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _require_torch():
    """Import and return torch modules."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        return torch, nn, F
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for GNN training. "
            "Install with: pip install torch>=2.0.0"
        ) from e


def _require_pyg():
    """Import and return PyG modules."""
    try:
        import torch_geometric
        from torch_geometric.data import Data, Batch
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, global_mean_pool
        return torch_geometric, Data, Batch, DataLoader, GCNConv, SAGEConv, GATv2Conv, global_mean_pool
    except ImportError as e:
        raise ImportError(
            "torch-geometric is required for GNN training. "
            "Install with: pip install torch-geometric>=2.4.0"
        ) from e


def _load_dataset_from_csv(csv_dir: str, feature_subset: list[str] | None = None):
    """
    Load graphs/nodes/edges CSVs and build PyG Data objects.

    Args:
        csv_dir: Directory containing graphs.csv, nodes.csv, edges.csv.
        feature_subset: Optional list of feature column names to use.
            If None, use all feat_* columns.

    Returns:
        list of PyG Data objects, one per graph.
    """
    torch, _, _ = _require_torch()
    _, Data, _, _, _, _, _, _ = _require_pyg()

    graphs_df = pd.read_csv(os.path.join(csv_dir, "graphs.csv"))
    nodes_df = pd.read_csv(os.path.join(csv_dir, "nodes.csv"))
    edges_df = pd.read_csv(os.path.join(csv_dir, "edges.csv"))

    feat_cols = feature_subset or [c for c in nodes_df.columns if c.startswith("feat_")]

    data_list = []
    for _, g_row in graphs_df.iterrows():
        gid = g_row["graph_id"]
        g_nodes = nodes_df[nodes_df["graph_id"] == gid].sort_values("node_id")
        g_edges = edges_df[edges_df["graph_id"] == gid]

        # Node features
        x = torch.tensor(g_nodes[feat_cols].values, dtype=torch.float32)

        # Edge index
        if len(g_edges) > 0:
            src = torch.tensor(g_edges["src_id"].values, dtype=torch.long)
            dst = torch.tensor(g_edges["dst_id"].values, dtype=torch.long)
            edge_index = torch.stack([src, dst], dim=0)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Graph label
        y = torch.tensor([g_row["label"]], dtype=torch.long)

        # Split info
        split = g_row.get("split", "train")

        data = Data(x=x, edge_index=edge_index, y=y)
        data.split = split
        data_list.append(data)

    return data_list


class MEPGraphClassifier:
    """
    Graph-level classifier for MEP conformance prediction.

    Architecture:
    - N conv layers (GCNConv, SAGEConv, or GATv2Conv)
    - Optional batch norm after each conv
    - Global mean pooling
    - 2-layer MLP head -> binary classification
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.3,
        conv_type: str = "GCN",
    ):
        torch, nn, F = _require_torch()
        _, _, _, _, GCNConv, SAGEConv, GATv2Conv, global_mean_pool = _require_pyg()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.conv_type = conv_type

        # Build model
        self.model = self._build_model(in_channels, hidden_dim, n_layers, dropout, conv_type)

    def _build_model(self, in_channels, hidden_dim, n_layers, dropout, conv_type):
        torch, nn, F = _require_torch()
        _, _, _, _, GCNConv, SAGEConv, GATv2Conv, global_mean_pool = _require_pyg()

        conv_cls = {"GCN": GCNConv, "SAGE": SAGEConv, "GATv2": GATv2Conv}[conv_type]

        class _Model(nn.Module):
            def __init__(self_m):
                super().__init__()
                self_m.convs = nn.ModuleList()
                self_m.bns = nn.ModuleList()

                # First layer
                if conv_type == "GATv2":
                    self_m.convs.append(conv_cls(in_channels, hidden_dim, heads=1))
                else:
                    self_m.convs.append(conv_cls(in_channels, hidden_dim))
                self_m.bns.append(nn.BatchNorm1d(hidden_dim))

                # Hidden layers
                for _ in range(n_layers - 1):
                    if conv_type == "GATv2":
                        self_m.convs.append(conv_cls(hidden_dim, hidden_dim, heads=1))
                    else:
                        self_m.convs.append(conv_cls(hidden_dim, hidden_dim))
                    self_m.bns.append(nn.BatchNorm1d(hidden_dim))

                # MLP head
                self_m.lin1 = nn.Linear(hidden_dim, hidden_dim // 2)
                self_m.lin2 = nn.Linear(hidden_dim // 2, 2)
                self_m.dropout = dropout

            def forward(self_m, data):
                x, edge_index, batch = data.x, data.edge_index, data.batch

                for conv, bn in zip(self_m.convs, self_m.bns):
                    x = conv(x, edge_index)
                    x = bn(x)
                    x = F.relu(x)
                    x = F.dropout(x, p=self_m.dropout, training=self_m.training)

                x = global_mean_pool(x, batch)
                x = F.relu(self_m.lin1(x))
                x = F.dropout(x, p=self_m.dropout, training=self_m.training)
                x = self_m.lin2(x)
                return x

        return _Model()


class MEPConformanceTrainer:
    """
    K-fold cross-validation trainer for MEP graph classification.
    """

    def __init__(
        self,
        csv_dir: str,
        n_folds: int = 5,
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 32,
        feature_subset: list[str] | None = None,
    ):
        self.csv_dir = csv_dir
        self.n_folds = n_folds
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.feature_subset = feature_subset
        self.data_list = _load_dataset_from_csv(csv_dir, feature_subset)

    def train_fold(
        self,
        train_data: list,
        val_data: list,
        config: dict,
    ) -> dict:
        """Train one fold and return metrics."""
        torch, nn, F = _require_torch()
        _, _, _, DataLoader, _, _, _, _ = _require_pyg()

        in_channels = train_data[0].x.shape[1] if train_data else 1

        clf = MEPGraphClassifier(
            in_channels=in_channels,
            hidden_dim=config.get("hidden_dim", 64),
            n_layers=config.get("n_layers", 3),
            dropout=config.get("dropout", 0.3),
            conv_type=config.get("conv_type", "GCN"),
        )
        model = clf.model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

        best_val_acc = 0.0
        best_val_auroc = 0.0

        for epoch in range(self.epochs):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                out = model(batch)
                probs = torch.softmax(out, dim=1)[:, 1]
                preds = out.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        acc = (all_preds == all_labels).mean()

        # AUROC
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(all_labels)) > 1:
                auroc = roc_auc_score(all_labels, all_probs)
            else:
                auroc = 0.5
        except Exception:
            auroc = 0.5

        return {
            "accuracy": float(acc),
            "auroc": float(auroc),
            "predictions": all_preds.tolist(),
            "probabilities": all_probs.tolist(),
            "labels": all_labels.tolist(),
        }

    def run_kfold(self, config: dict | None = None) -> dict:
        """Run k-fold cross-validation."""
        if config is None:
            config = {}

        rng = np.random.default_rng(42)
        indices = np.arange(len(self.data_list))
        rng.shuffle(indices)

        fold_size = len(indices) // self.n_folds
        fold_results = []

        for fold in range(self.n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < self.n_folds - 1 else len(indices)
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

            train_data = [self.data_list[i] for i in train_idx]
            val_data = [self.data_list[i] for i in val_idx]

            result = self.train_fold(train_data, val_data, config)
            result["fold"] = fold
            fold_results.append(result)

        mean_acc = np.mean([r["accuracy"] for r in fold_results])
        mean_auroc = np.mean([r["auroc"] for r in fold_results])

        return {
            "folds": fold_results,
            "mean_accuracy": float(mean_acc),
            "mean_auroc": float(mean_auroc),
            "config": config,
        }


def run_hyperparameter_search(
    csv_dir: str,
    output_csv: str = "results/hyperparameter_search.csv",
    n_folds: int = 5,
    epochs: int = 50,
) -> pd.DataFrame:
    """
    Grid search over conv_type x hidden_dim x n_layers x dropout.
    81 configs x 5 folds = 405 runs.
    """
    conv_types = ["GCN", "SAGE", "GATv2"]
    hidden_dims = [32, 64, 128]
    n_layers_list = [2, 3, 4]
    dropouts = [0.1, 0.3, 0.5]

    results = []
    total = len(conv_types) * len(hidden_dims) * len(n_layers_list) * len(dropouts)
    run_idx = 0

    for conv_type in conv_types:
        for hidden_dim in hidden_dims:
            for n_layers in n_layers_list:
                for dropout in dropouts:
                    run_idx += 1
                    logger.info(
                        "Hyperparameter search %d/%d: %s h=%d l=%d d=%.1f",
                        run_idx, total, conv_type, hidden_dim, n_layers, dropout,
                    )

                    config = {
                        "conv_type": conv_type,
                        "hidden_dim": hidden_dim,
                        "n_layers": n_layers,
                        "dropout": dropout,
                    }

                    trainer = MEPConformanceTrainer(
                        csv_dir=csv_dir,
                        n_folds=n_folds,
                        epochs=epochs,
                    )
                    cv_result = trainer.run_kfold(config)

                    for fold_result in cv_result["folds"]:
                        results.append({
                            "conv_type": conv_type,
                            "hidden_dim": hidden_dim,
                            "n_layers": n_layers,
                            "dropout": dropout,
                            "fold": fold_result["fold"],
                            "accuracy": fold_result["accuracy"],
                            "auroc": fold_result["auroc"],
                        })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Hyperparameter search results saved to %s (%d rows).", output_csv, len(df))
    return df


def run_ablation_study(
    csv_dir: str,
    best_config: dict,
    output_csv: str = "results/ablation_study.csv",
    n_folds: int = 5,
    epochs: int = 50,
) -> pd.DataFrame:
    """
    Run the best config with four feature subsets:
    topology_only, geometry, hydraulic, full.
    """
    feature_subsets = {
        "topology_only": [
            "feat_ifc_type_0", "feat_ifc_type_1", "feat_ifc_type_2",
            "feat_ifc_type_3", "feat_ifc_type_4",
            "feat_depth_from_root", "feat_is_terminal",
            "feat_betweenness_centrality",
        ],
        "geometry": [
            "feat_ifc_type_0", "feat_ifc_type_1", "feat_ifc_type_2",
            "feat_ifc_type_3", "feat_ifc_type_4",
            "feat_depth_from_root", "feat_is_terminal",
            "feat_betweenness_centrality",
            "feat_elevation_m",
        ],
        "hydraulic": [
            "feat_ifc_type_0", "feat_ifc_type_1", "feat_ifc_type_2",
            "feat_ifc_type_3", "feat_ifc_type_4",
            "feat_depth_from_root", "feat_is_terminal",
            "feat_betweenness_centrality",
            "feat_nominal_diameter_mm", "feat_length_m",
        ],
        "full": None,  # All features
    }

    results = []
    for subset_name, feat_cols in feature_subsets.items():
        logger.info("Ablation: %s", subset_name)
        trainer = MEPConformanceTrainer(
            csv_dir=csv_dir,
            n_folds=n_folds,
            epochs=epochs,
            feature_subset=feat_cols,
        )
        cv_result = trainer.run_kfold(best_config)

        for fold_result in cv_result["folds"]:
            results.append({
                "feature_subset": subset_name,
                "fold": fold_result["fold"],
                "accuracy": fold_result["accuracy"],
                "auroc": fold_result["auroc"],
                **best_config,
            })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Ablation study results saved to %s.", output_csv)
    return df


def run_random_forest_baseline(csv_dir: str) -> dict:
    """Random Forest baseline on graph-level features."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score

    graphs_df = pd.read_csv(os.path.join(csv_dir, "graphs.csv"))
    nodes_df = pd.read_csv(os.path.join(csv_dir, "nodes.csv"))

    feat_cols = [c for c in nodes_df.columns if c.startswith("feat_")]

    # Aggregate node features per graph
    graph_feats = nodes_df.groupby("graph_id")[feat_cols].agg(["mean", "std", "max", "min"])
    graph_feats.columns = ["_".join(col) for col in graph_feats.columns]
    graph_feats = graph_feats.fillna(0)

    merged = graphs_df.merge(graph_feats, left_on="graph_id", right_index=True)

    train = merged[merged["split"] == "train"]
    test = merged[merged["split"] == "test"]

    if len(test) == 0:
        # Use val as test fallback
        test = merged[merged["split"] == "val"]

    feature_cols = [c for c in graph_feats.columns]
    X_train = train[feature_cols].values
    y_train = train["label"].values
    X_test = test[feature_cols].values
    y_test = test["label"].values

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)

    acc = accuracy_score(y_test, preds)
    try:
        auroc = roc_auc_score(y_test, probs[:, 1]) if len(np.unique(y_test)) > 1 else 0.5
    except Exception:
        auroc = 0.5

    return {"model": "RandomForest", "accuracy": acc, "auroc": auroc}


def run_mlp_baseline(csv_dir: str) -> dict:
    """MLP baseline on graph-level features."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score

    graphs_df = pd.read_csv(os.path.join(csv_dir, "graphs.csv"))
    nodes_df = pd.read_csv(os.path.join(csv_dir, "nodes.csv"))

    feat_cols = [c for c in nodes_df.columns if c.startswith("feat_")]

    graph_feats = nodes_df.groupby("graph_id")[feat_cols].agg(["mean", "std", "max", "min"])
    graph_feats.columns = ["_".join(col) for col in graph_feats.columns]
    graph_feats = graph_feats.fillna(0)

    merged = graphs_df.merge(graph_feats, left_on="graph_id", right_index=True)

    train = merged[merged["split"] == "train"]
    test = merged[merged["split"] == "test"]

    if len(test) == 0:
        test = merged[merged["split"] == "val"]

    feature_cols = [c for c in graph_feats.columns]
    X_train = train[feature_cols].values
    y_train = train["label"].values
    X_test = test[feature_cols].values
    y_test = test["label"].values

    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=500,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)

    acc = accuracy_score(y_test, preds)
    try:
        auroc = roc_auc_score(y_test, probs[:, 1]) if len(np.unique(y_test)) > 1 else 0.5
    except Exception:
        auroc = 0.5

    return {"model": "MLP", "accuracy": acc, "auroc": auroc}

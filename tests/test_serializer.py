"""Tests for serializer.py — CSV export for PyG."""
from __future__ import annotations

import os
import pandas as pd
import pytest
from ifc_hydro_topologic.hydro_generator import generate_corpus
from ifc_hydro_topologic.serializer import serialize_dataset_to_pyg_csv


@pytest.fixture
def small_corpus():
    """Generate a small corpus for testing serialization."""
    return generate_corpus(n=20, seed=99)


class TestSerializeDataset:
    def test_creates_three_csv_files(self, small_corpus, tmp_path):
        """graphs.csv, nodes.csv, edges.csv all created."""
        result = serialize_dataset_to_pyg_csv(small_corpus, str(tmp_path))
        assert os.path.exists(result["graphs_csv"])
        assert os.path.exists(result["nodes_csv"])
        assert os.path.exists(result["edges_csv"])

    def test_graph_count_matches(self, small_corpus, tmp_path):
        """Number of rows in graphs.csv equals corpus size."""
        serialize_dataset_to_pyg_csv(small_corpus, str(tmp_path))
        df = pd.read_csv(tmp_path / "graphs.csv")
        assert len(df) == len(small_corpus)

    def test_node_features_no_nan(self, small_corpus, tmp_path):
        """All feat_* columns are non-NaN."""
        serialize_dataset_to_pyg_csv(small_corpus, str(tmp_path))
        df = pd.read_csv(tmp_path / "nodes.csv")
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        assert len(feat_cols) > 0
        for col in feat_cols:
            assert df[col].isna().sum() == 0, f"NaN found in {col}"

    def test_class_balance_keys(self, small_corpus, tmp_path):
        """Returned class_balance has keys 0 and 1."""
        result = serialize_dataset_to_pyg_csv(small_corpus, str(tmp_path))
        assert 0 in result["class_balance"]
        assert 1 in result["class_balance"]

    def test_node_label_values(self, small_corpus, tmp_path):
        """Node labels are -1, 0, or 1."""
        serialize_dataset_to_pyg_csv(small_corpus, str(tmp_path))
        df = pd.read_csv(tmp_path / "nodes.csv")
        assert set(df["label"].unique()).issubset({-1, 0, 1})

    def test_train_val_test_masks_sum(self, small_corpus, tmp_path):
        """For terminal nodes, exactly one mask is True."""
        serialize_dataset_to_pyg_csv(small_corpus, str(tmp_path))
        df = pd.read_csv(tmp_path / "nodes.csv")
        terminals = df[df["feat_is_terminal"] == 1]
        mask_sum = terminals["train_mask"].astype(int) + terminals["val_mask"].astype(int) + terminals["test_mask"].astype(int)
        assert (mask_sum == 1).all(), "Each terminal should have exactly one mask set"

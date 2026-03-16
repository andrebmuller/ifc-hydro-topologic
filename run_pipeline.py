"""Run the full pipeline: generate corpus, serialize, train GNN, hyperparameter search, ablation, baselines."""
from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
CSV_DIR = os.path.join(RESULTS_DIR, "csv_data")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)


def step1_generate_corpus():
    """Generate N=500 synthetic plumbing graphs."""
    from ifc_hydro_topologic.hydro_generator import generate_corpus
    logger.info("=== Step 1: Generating corpus (N=500) ===")
    t0 = time.time()
    corpus = generate_corpus(n=500, seed=42, target_nonconforming_fraction=0.40)
    t1 = time.time()
    nc = sum(1 for _, l in corpus if not l["system_conforms"])
    logger.info("Generated %d graphs in %.1fs. NC fraction: %.2f", len(corpus), t1 - t0, nc / len(corpus))
    return corpus


def step2_serialize(corpus):
    """Serialize corpus to CSV for PyG."""
    from ifc_hydro_topologic.serializer import serialize_dataset_to_pyg_csv
    logger.info("=== Step 2: Serializing to CSV ===")
    result = serialize_dataset_to_pyg_csv(corpus, CSV_DIR)
    logger.info("Serialized: %s", result)
    return result


def step3_hyperparameter_search():
    """Run hyperparameter grid search (81 configs x 5 folds)."""
    from ifc_hydro_topologic.gnn import run_hyperparameter_search
    logger.info("=== Step 3: Hyperparameter search ===")
    t0 = time.time()
    df = run_hyperparameter_search(
        CSV_DIR,
        output_csv=os.path.join(RESULTS_DIR, "hyperparameter_search.csv"),
        n_folds=5,
        epochs=50,
    )
    t1 = time.time()
    logger.info("Hyperparameter search done in %.1fs. %d rows.", t1 - t0, len(df))

    # Find best config
    mean_scores = df.groupby(["conv_type", "hidden_dim", "n_layers", "dropout"])["auroc"].mean()
    best_idx = mean_scores.idxmax()
    best_auroc = mean_scores.max()
    logger.info("Best config: %s with mean AUROC=%.4f", best_idx, best_auroc)

    best_config = {
        "conv_type": best_idx[0],
        "hidden_dim": int(best_idx[1]),
        "n_layers": int(best_idx[2]),
        "dropout": float(best_idx[3]),
    }
    return df, best_config


def step4_ablation(best_config):
    """Run ablation study with best config."""
    from ifc_hydro_topologic.gnn import run_ablation_study
    logger.info("=== Step 4: Ablation study ===")
    df = run_ablation_study(
        CSV_DIR,
        best_config,
        output_csv=os.path.join(RESULTS_DIR, "ablation_study.csv"),
        n_folds=5,
        epochs=50,
    )
    logger.info("Ablation done. %d rows.", len(df))
    return df


def step5_baselines():
    """Run RF and MLP baselines."""
    from ifc_hydro_topologic.gnn import run_random_forest_baseline, run_mlp_baseline
    logger.info("=== Step 5: Baselines ===")
    rf = run_random_forest_baseline(CSV_DIR)
    mlp = run_mlp_baseline(CSV_DIR)
    logger.info("RF: acc=%.4f, auroc=%.4f", rf["accuracy"], rf["auroc"])
    logger.info("MLP: acc=%.4f, auroc=%.4f", mlp["accuracy"], mlp["auroc"])

    df = pd.DataFrame([rf, mlp])
    df.to_csv(os.path.join(RESULTS_DIR, "baseline_comparison.csv"), index=False)
    return df


def step6_predictions(best_config):
    """Generate test set predictions with best config."""
    from ifc_hydro_topologic.gnn import MEPConformanceTrainer
    logger.info("=== Step 6: Generating predictions ===")

    trainer = MEPConformanceTrainer(csv_dir=CSV_DIR, n_folds=5, epochs=50)
    cv_result = trainer.run_kfold(best_config)

    # Collect all fold predictions
    all_rows = []
    for fold_result in cv_result["folds"]:
        for pred, prob, label in zip(
            fold_result["predictions"],
            fold_result["probabilities"],
            fold_result["labels"],
        ):
            all_rows.append({
                "fold": fold_result["fold"],
                "true_label": label,
                "predicted_label": pred,
                "probability_conform": prob,
            })

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "predictions.csv"), index=False)
    logger.info("Predictions saved: %d rows, mean AUROC=%.4f", len(df), cv_result["mean_auroc"])
    return df


if __name__ == "__main__":
    corpus = step1_generate_corpus()
    step2_serialize(corpus)
    hp_df, best_config = step3_hyperparameter_search()
    step4_ablation(best_config)
    step5_baselines()
    step6_predictions(best_config)
    logger.info("=== Pipeline complete! ===")

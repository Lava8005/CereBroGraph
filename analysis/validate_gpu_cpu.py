"""Empirical Validation: Establishes a CPU noise baseline to evaluate GPU variance."""

import numpy as np
import logging
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(level=logging.INFO, format="%(message)s")

def _get_top_k_overlap(dict1: dict, dict2: dict, top_percent: float = 0.10) -> float:
    k = max(1, int(len(dict1) * top_percent))
    top1 = set(sorted(dict1.keys(), key=lambda x: dict1[x])[:k])
    top2 = set(sorted(dict2.keys(), key=lambda x: dict2[x])[:k])
    return len(top1.intersection(top2)) / k

def run_validation(cpu_seed1: dict, cpu_seed2: dict, gpu_results: dict):
    shared = set(cpu_seed1.keys()) & set(cpu_seed2.keys()) & set(gpu_results.keys())
    if len(shared) < 50:
        raise ValueError("Require at least 50 overlapping ligands for statistical significance.")

    c1 = np.array([cpu_seed1[k] for k in shared])
    c2 = np.array([cpu_seed2[k] for k in shared])
    g1 = np.array([gpu_results[k] for k in shared])

    # CPU vs CPU (Baseline Noise)
    cpu_mae = np.mean(np.abs(c1 - c2))
    cpu_spearman, _ = spearmanr(c1, c2)
    cpu_overlap = _get_top_k_overlap({k: cpu_seed1[k] for k in shared}, {k: cpu_seed2[k] for k in shared})

    # CPU vs GPU
    gpu_mae = np.mean(np.abs(c1 - g1))
    gpu_spearman, _ = spearmanr(c1, g1)
    gpu_overlap = _get_top_k_overlap({k: cpu_seed1[k] for k in shared}, {k: gpu_results[k] for k in shared})

    logging.info(f"Sample Size: {len(shared)} ligands")
    logging.info("-" * 40)
    logging.info("BASELINE (CPU Seed A vs CPU Seed B):")
    logging.info(f"  Mean Absolute Error : {cpu_mae:.4f} kcal/mol")
    logging.info(f"  Spearman Rank       : {cpu_spearman:.4f}")
    logging.info(f"  Top-10% Overlap     : {cpu_overlap:.2%}")
    logging.info("-" * 40)
    logging.info("EVALUATION (CPU Seed A vs GPU OpenCL):")
    logging.info(f"  Mean Absolute Error : {gpu_mae:.4f} kcal/mol")
    logging.info(f"  Spearman Rank       : {gpu_spearman:.4f}")
    logging.info(f"  Top-10% Overlap     : {gpu_overlap:.2%}")
    logging.info("-" * 40)
    
    if gpu_overlap < (cpu_overlap * 0.8):
        logging.warning("⚠️ GPU Top-10% overlap diverges significantly from the CPU baseline.")
        logging.warning("The Pareto front will physical shift. Acknowledge this variance in publication.")
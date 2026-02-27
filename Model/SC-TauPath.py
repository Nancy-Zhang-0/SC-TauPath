#!/usr/bin/env python3
"""
Pairwise SC-Tau Attribution Analysis (Main Result)
===================================================
Framework: ΔSC_ij → Δτ_ij (Ridge regression in PCA space)

Three levels of analysis:
  1. Overall:     fit on all pairs → global edge importance + hub ROIs
  2. Pair-type:   fit separate Ridge per pair type (CN-AD, CN-CN, MCI-AD, AD-AD …)
                  → which connections drive tau differences in each group comparison
  3. Stability:   cross-fold Jaccard for both overall and per-pair-type importance

Attribution pipeline (fully coefficient-based, no gradient approximation):
  PCA fit on fit-train SC features
  Ridge fit on pairwise SC differences (PCA space)
  coef_ back-projected: PCA.components_.T @ coef_pca → (n_sc_feat, n_nodes)
  edge_importance[k] = Σ_r |coef[edge_k, roi_r]|   (sum over all 246 tau ROIs)
  hub_score[r] = Σ_{k: r∈edge_k} edge_importance[k]  (degree-weighted)

Outputs (all in --out-dir):
  pairwise_attr_fold_metrics.csv       -- per-fold prediction metrics
  pairwise_attr_oof_predictions.npz    -- OOF pair-level predictions
  pairwise_attr_edge_importance.csv    -- all edges, overall importance (agg across folds)
  pairwise_attr_edge_topk.csv          -- top-K edges with ROI labels
  pairwise_attr_hub_roi_scores.csv     -- hub ROI scores + mean tau
  pairwise_attr_pair_type_metrics.csv  -- OOF metrics per pair type
  pairwise_attr_pair_type_edges.csv    -- top-K edges per pair type
  pairwise_attr_pair_type_hubs.csv     -- hub scores per pair type
  pairwise_attr_stability.json         -- Jaccard stability across folds
  pairwise_attr_meta.json              -- full config + summary
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge
from sklearn.multioutput import MultiOutputRegressor

try:
    from quick_train_tau_mlp_gpu import (
        build_full_network_features,
        fit_ndm_global_params,
        ndm_predict_from_eig,
        ndm_precompute_eig,
        parse_float_list_arg,
        parse_int_list_arg,
        parse_list_arg,
        set_seed,
    )
except ImportError:
    from Model.quick_train_tau_mlp_gpu import (
        build_full_network_features,
        fit_ndm_global_params,
        ndm_predict_from_eig,
        ndm_precompute_eig,
        parse_float_list_arg,
        parse_int_list_arg,
        parse_list_arg,
        set_seed,
    )

LABEL_MAP = {"CN": 0, "MCI": 1, "AD": 2}
PAIR_TYPE_NAMES = {
    (0, 0): "CN-CN",
    (0, 1): "CN-MCI",
    (0, 2): "CN-AD",
    (1, 1): "MCI-MCI",
    (1, 2): "MCI-AD",
    (2, 2): "AD-AD",
}
# Minimum pairs needed to fit a reliable stratified model
MIN_PAIRS_FOR_STRATIFIED = 200


# ============================================================
# Utilities
# ============================================================

def load_diagnosis_map(pair_csv: str) -> Dict[str, str]:
    out = {}
    with open(pair_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = (row.get("subject_id") or "").strip()
            dx  = (row.get("pet_diagnosis_label") or "").strip().upper()
            if sid:
                out[sid] = dx
    return out


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(pearsonr(a, b)[0])


def mean_pearson_by_pair(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean per-pair Pearson (each pair = 246-dim tau difference vector).
    Vectorized implementation for speed and stable handling of near-constant rows.
    """
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"mean_pearson_by_pair expects (n_pairs, n_nodes) arrays with same shape, got {a.shape} vs {b.shape}")

    a_c = a - np.mean(a, axis=1, keepdims=True)
    b_c = b - np.mean(b, axis=1, keepdims=True)
    a_n = np.sqrt(np.sum(a_c * a_c, axis=1))
    b_n = np.sqrt(np.sum(b_c * b_c, axis=1))
    den = a_n * b_n
    valid = den > 1e-12
    if not np.any(valid):
        return float("nan")
    num = np.sum(a_c * b_c, axis=1)
    return float(np.mean(num[valid] / den[valid]))


def permutation_test_mean_pearson_by_pair(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Permutation test for H1: observed mean per-pair Pearson is greater than random pairing.
    Null is created by permuting prediction rows across pairs.
    """
    if n_perm <= 0:
        return {
            "observed_r": float("nan"),
            "p_value_gt_random": float("nan"),
            "null_mean": float("nan"),
            "null_std": float("nan"),
            "n_perm": 0,
            "n_perm_valid": 0,
        }

    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"permutation_test_mean_pearson_by_pair expects 2D arrays with same shape, got {a.shape} vs {b.shape}")

    n_pairs = a.shape[0]
    if n_pairs < 2:
        obs = mean_pearson_by_pair(a, b)
        return {
            "observed_r": float(obs),
            "p_value_gt_random": float("nan"),
            "null_mean": float("nan"),
            "null_std": float("nan"),
            "n_perm": int(n_perm),
            "n_perm_valid": 0,
        }

    a_c = a - np.mean(a, axis=1, keepdims=True)
    b_c = b - np.mean(b, axis=1, keepdims=True)
    a_n = np.sqrt(np.sum(a_c * a_c, axis=1))
    b_n = np.sqrt(np.sum(b_c * b_c, axis=1))

    obs_den = a_n * b_n
    obs_valid = obs_den > 1e-12
    if not np.any(obs_valid):
        obs_r = float("nan")
    else:
        obs_num = np.sum(a_c * b_c, axis=1)
        obs_r = float(np.mean(obs_num[obs_valid] / obs_den[obs_valid]))

    null_rs = np.full(n_perm, np.nan, dtype=np.float64)
    for t in range(n_perm):
        perm = rng.permutation(n_pairs)
        den = a_n * b_n[perm]
        valid = den > 1e-12
        if not np.any(valid):
            continue
        num = np.sum(a_c * b_c[perm], axis=1)
        null_rs[t] = float(np.mean(num[valid] / den[valid]))

    null_valid = null_rs[~np.isnan(null_rs)]
    if len(null_valid) == 0 or np.isnan(obs_r):
        p_gt = float("nan")
    else:
        # One-sided exact p-value with +1 correction: H1 = observed r > random
        p_gt = float((1 + np.sum(null_valid >= obs_r)) / (len(null_valid) + 1))

    return {
        "observed_r": float(obs_r),
        "p_value_gt_random": p_gt,
        "null_mean": float(np.mean(null_valid)) if len(null_valid) else float("nan"),
        "null_std": float(np.std(null_valid)) if len(null_valid) else float("nan"),
        "n_perm": int(n_perm),
        "n_perm_valid": int(len(null_valid)),
    }


def evaluate_pairs(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt, yp = y_true.reshape(-1), y_pred.reshape(-1)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    return {
        "r2_flat":        float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "mse":            float(np.mean((yt - yp) ** 2)),
        "pearson_by_pair": mean_pearson_by_pair(y_true, y_pred),
        "n_pairs":        int(y_true.shape[0]),
    }


def build_subject_kfold(n_subj: int, n_folds: int, seed: int):
    rng   = np.random.default_rng(seed)
    perm  = rng.permutation(n_subj)
    sizes = np.full(n_folds, n_subj // n_folds, dtype=int)
    sizes[: n_subj % n_folds] += 1
    splits, start = [], 0
    for sz in sizes:
        test  = perm[start : start + sz]
        train = np.concatenate([perm[:start], perm[start + sz:]])
        splits.append((train, test))
        start += sz
    return splits


def build_pairs(
    feat:     np.ndarray,   # (n_subj, n_feat) PCA space
    tau:      np.ndarray,   # (n_subj, n_nodes)
    dx_code:  np.ndarray,
    subj_idx: np.ndarray,
    augment:  bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build pairwise dataset from a subset of subjects."""
    n = len(subj_idx)
    if augment:
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        mask   = (ii != jj)
        ii, jj = ii[mask].ravel(), jj[mask].ravel()
    else:
        ii, jj = np.triu_indices(n, k=1)

    gi, gj   = subj_idx[ii], subj_idx[jj]
    X_diff   = (feat[gi] - feat[gj]).astype(np.float32)
    y_diff   = (tau[gi]  - tau[gj]).astype(np.float32)
    dx_pair  = np.stack([
        np.minimum(dx_code[gi], dx_code[gj]),
        np.maximum(dx_code[gi], dx_code[gj]),
    ], axis=1)
    return X_diff, y_diff, gi, gj, dx_pair


def make_ridge(alpha: float) -> Ridge:
    """
    Use solver='lsqr' (iterative, no matrix inversion) to avoid
    ill-conditioned LinAlgWarning that occurs with the default Cholesky
    solver when ΔSC pairwise features have rcond ~1e-9.
    """
    return Ridge(alpha=alpha, solver="lsqr", max_iter=10000)


def fit_ridge_select_alpha(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_vl: np.ndarray, y_vl: np.ndarray,
    alphas: List[float],
) -> Tuple[MultiOutputRegressor, float, float]:
    """Fit Ridge with alpha selected by validation Pearson."""
    best_score, best_alpha = -float("inf"), alphas[0]
    for alpha in alphas:
        m = MultiOutputRegressor(make_ridge(alpha), n_jobs=-1)
        m.fit(X_tr, y_tr)
        if X_vl.shape[0] > 0:
            sc = mean_pearson_by_pair(y_vl, m.predict(X_vl).astype(np.float32))
            if not np.isnan(sc) and sc > best_score:
                best_score, best_alpha = sc, alpha
    model = MultiOutputRegressor(make_ridge(best_alpha), n_jobs=-1)
    model.fit(X_tr, y_tr)
    return model, best_alpha, best_score


def fit_pls_select_ncomp(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_vl: np.ndarray, y_vl: np.ndarray,
    n_comp_grid: List[int],
) -> Tuple[PLSRegression, int, float]:
    """Fit PLSRegression selecting n_components by validation Pearson."""
    best_score, best_n = -float("inf"), n_comp_grid[0]
    for n in n_comp_grid:
        n = min(n, X_tr.shape[1], X_tr.shape[0] - 1)
        m = PLSRegression(n_components=n, max_iter=1000)
        m.fit(X_tr, y_tr)
        if X_vl.shape[0] > 0:
            sc = mean_pearson_by_pair(y_vl, m.predict(X_vl).astype(np.float32))
            if not np.isnan(sc) and sc > best_score:
                best_score, best_n = sc, n
    best_n = min(best_n, X_tr.shape[1], X_tr.shape[0] - 1)
    model = PLSRegression(n_components=best_n, max_iter=1000)
    model.fit(X_tr, y_tr)
    return model, best_n, best_score


def fit_lasso_select_alpha(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_vl: np.ndarray, y_vl: np.ndarray,
    alphas: List[float],
) -> Tuple[MultiOutputRegressor, float, float]:
    """Fit Lasso (MultiOutput) selecting alpha by validation Pearson."""
    best_score, best_alpha = -float("inf"), alphas[0]
    for alpha in alphas:
        m = MultiOutputRegressor(
            Lasso(alpha=alpha, max_iter=10000, tol=1e-4), n_jobs=-1
        )
        m.fit(X_tr, y_tr)
        if X_vl.shape[0] > 0:
            sc = mean_pearson_by_pair(y_vl, m.predict(X_vl).astype(np.float32))
            if not np.isnan(sc) and sc > best_score:
                best_score, best_alpha = sc, alpha
    model = MultiOutputRegressor(
        Lasso(alpha=best_alpha, max_iter=10000, tol=1e-4), n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    return model, best_alpha, best_score


def pls_coef_to_edge_importance(
    model:    PLSRegression,
    pca:      PCA,
    n_edges:  int,
    n_nodes:  int,
    use_edge_keys: List[str],
) -> np.ndarray:
    """
    Extract edge importance from PLSRegression.
    PLS coef_ shape: (n_pca [+n_ndm], n_nodes).
    Back-project SC-PCA portion same way as Ridge.
    """
    n_pca = pca.n_components_
    if n_pca == 0:
        return np.zeros(n_edges, dtype=np.float64)
    coef_full = model.coef_.T  # sklearn PLS: coef_ is (n_targets, n_features) → transpose
    coef_pca  = coef_full[:n_pca, :]
    coef_orig = pca.components_.T @ coef_pca
    n_ec = len(use_edge_keys)
    imp  = np.zeros(n_edges, dtype=np.float64)
    for c in range(n_ec):
        s = c * n_edges
        e = (c + 1) * n_edges
        imp += np.sum(np.abs(coef_orig[s:e, :]), axis=1)
    return imp


# ============================================================
# Attribution: coefficient → edge importance
# ============================================================

def coef_to_edge_importance(
    model:     MultiOutputRegressor,
    pca:       PCA,
    n_edges:   int,
    n_nodes:   int,
    use_edge_keys: List[str],
) -> np.ndarray:
    """
    Extract edge importance from Ridge model coefficients.

    When NDM features are appended after PCA features, coef_ has shape
    (n_pca + n_ndm, ). We slice only the first n_pca dims for back-projection;
    the NDM portion is intentionally excluded (it's a node-level feature,
    not an edge attribution target).

    Steps:
      1. Stack per-output Ridge coefs → coef_full (n_pca [+ n_ndm], n_nodes)
      2. Slice SC-PCA part: coef_pca = coef_full[:n_pca, :]
      3. Back-project: coef_orig = PCA.components_.T @ coef_pca  (n_sc_feat, n_nodes)
      4. edge_imp[k] = Σ_c Σ_r |coef_orig[c*n_edges + k, r]|

    Returns:
        edge_importance: (n_edges,) aggregated importance
    """
    n_pca = pca.n_components_
    # coef_ shape per estimator: (n_pca,) or (n_pca + n_ndm,) or (n_ndm,) for ndm-only
    coef_full = np.stack([est.coef_ for est in model.estimators_], axis=1)  # (n_pca [+n_ndm], n_nodes)
    if n_pca == 0:
        # NDM-only mode: no SC features → edge importance is all zeros
        return np.zeros(n_edges, dtype=np.float64)
    coef_pca  = coef_full[:n_pca, :]                                         # (n_pca, n_nodes)
    coef_orig = pca.components_.T @ coef_pca                                 # (n_sc_feat, n_nodes)

    n_ec   = len(use_edge_keys)
    imp    = np.zeros(n_edges, dtype=np.float64)
    for c in range(n_ec):
        s = c * n_edges
        e = (c + 1) * n_edges
        imp += np.sum(np.abs(coef_orig[s:e, :]), axis=1)
    return imp


def compute_hub_scores(
    edge_importance: np.ndarray,
    n_nodes: int,
    tri: Tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Hub score = sum of edge importance for all edges incident to each ROI."""
    hub = np.zeros(n_nodes, dtype=np.float64)
    for k in range(len(edge_importance)):
        i, j = int(tri[0][k]), int(tri[1][k])
        hub[i] += edge_importance[k]
        hub[j] += edge_importance[k]
    hub /= (hub.max() + 1e-12)
    return hub


def mean_jaccard(fold_sets: List[set]) -> float:
    js = []
    n  = len(fold_sets)
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(fold_sets[i] & fold_sets[j])
            union = len(fold_sets[i] | fold_sets[j])
            js.append(inter / union if union > 0 else 0.0)
    return float(np.mean(js)) if js else float("nan")


def write_csv(path: str, rows: List[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description="Pairwise SC-tau attribution: edge importance, hub ROIs, pair-type stratification"
    )
    p.add_argument("--data",      default="/mnt/disk3/ADNI_DTI_fMRI/Ana_Code/Model/GNN_Input/brainnectome_gnn_pet_tau_suvr.npz")
    p.add_argument("--pair-csv",  default="/mnt/disk3/ADNI_DTI_fMRI/Ana_Code/PET_Axial_DTI_one_pair_per_subject.csv")
    p.add_argument("--out-dir",   default="/mnt/disk3/ADNI_DTI_fMRI/Ana_Code/Model/GNN_Input/pairwise_attribution")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--cv-folds",   type=int,   default=5)
    p.add_argument("--val-ratio",  type=float, default=0.15)
    p.add_argument("--edge-keys",  default="A_count")
    p.add_argument("--node-keys",  default="strength_count,mean_fa_nonzero,mean_rd_nonzero")
    p.add_argument("--pca-components", type=int, default=0,
                   help="0 = full rank of fit-train (~158 dims). Explicit value overrides.")
    p.add_argument("--ridge-alphas",   default="100.0,1000.0,10000.0,100000.0,1000000.0",
                   help="Ridge alpha grid. Needs to be large for pairwise ΔSC features "
                        "(ill-conditioned matrix; rcond ~1e-9 typical).")
    p.add_argument("--topk-edges",     type=int, default=200)
    p.add_argument("--no-augment",     action="store_true")
    p.add_argument("--pls-ncomp-grid", default="10,20,50,80,120,158",
                   help="PLS n_components grid to search.")
    p.add_argument("--lasso-alphas",   default="10000.0,100000.0,1000000.0,10000000.0",
                   help="Lasso alpha grid (sparse; needs large alpha like Ridge).")
    p.add_argument("--compare-models",  action="store_true",
                   help="Also fit PLS and Lasso for model comparison. Off by default (slow).")
    p.add_argument("--perm-test", action="store_true",
                   help="Run one-sided permutation test for OOF Ridge Pearson (H1: r > random pairing).")
    p.add_argument("--n-perm", type=int, default=1000,
                   help="Number of permutations for --perm-test.")
    p.add_argument("--perm-seed", type=int, default=None,
                   help="Random seed for permutation test. Default: seed + 2026.")
    p.add_argument("--perm-test-pair-types", action="store_true",
                   help="Also run permutation test for each OOF pair type (slower).")
    # NDM
    p.add_argument("--ndm-enable",     action="store_true",
                   help="Add per-subject NDM diffusion features alongside SC-PCA features.")
    p.add_argument("--ndm-only",       action="store_true",
                   help="Use ONLY NDM diffusion features (no SC-PCA). "
                        "Ablation: tests whether diffusion physics alone predicts tau differences. "
                        "Automatically implies --ndm-enable.")
    p.add_argument("--ndm-edge-key",   default="A_count")
    p.add_argument("--ndm-seed-ids",   default="115,116",
                   help="1-based ROI IDs for NDM seed (default: entorhinal cortex).")
    p.add_argument("--ndm-beta-grid",  default="0.05,0.1,0.2,0.5,1.0")
    p.add_argument("--ndm-time-grid",  default="0.5,1.0,2.0,4.0,8.0")
    p.add_argument("--ndm-log1p-w",    action="store_true")
    p.add_argument("--ndm-residual",    action="store_true",
                   help="Residual mode: NDM predicts population-level tau baseline; "
                        "Ridge learns to predict Δ(tau - tau_NDM) from ΔSC. "
                        "NDM is the physics decomposer, SC captures individual residuals. "
                        "Automatically implies --ndm-enable. ΔNDM is NOT concatenated to input.")
    args = p.parse_args()

    # --ndm-only and --ndm-residual both imply --ndm-enable
    if args.ndm_only or args.ndm_residual:
        args.ndm_enable = True
    if args.ndm_only and args.ndm_residual:
        raise ValueError("--ndm-only and --ndm-residual are mutually exclusive.")

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    d = np.load(args.data, allow_pickle=True)
    subject_ids = np.array([str(x) for x in d["subject_ids"].tolist()])
    A  = d["A"].astype(np.float32)
    X  = d["X"].astype(np.float32)
    y  = d["y_suvr"].astype(np.float32)
    edge_names = [str(x) for x in d["edge_feature_names"].tolist()]
    node_names = [str(x) for x in d["node_feature_names"].tolist()]

    dx_map      = load_diagnosis_map(args.pair_csv)
    dx_str_all  = np.array([dx_map.get(sid, "") for sid in subject_ids], dtype=object)
    dx_code_all = np.array([LABEL_MAP.get(str(dx), -1) for dx in dx_str_all], dtype=np.int32)

    use_edge_keys = parse_list_arg(args.edge_keys)
    use_node_keys = parse_list_arg(args.node_keys)

    feat_all = build_full_network_features(
        A=A, X=X,
        edge_names=edge_names, node_names=node_names,
        use_edge_keys=use_edge_keys, use_node_keys=use_node_keys,
    )
    n_subj    = int(A.shape[0])
    n_nodes   = int(y.shape[1])
    n_sc_feat = feat_all.shape[1]
    n_edges   = (n_nodes * (n_nodes - 1)) // 2
    tri       = np.triu_indices(n_nodes, k=1)
    augment   = not args.no_augment
    ridge_alphas  = [float(x) for x in args.ridge_alphas.split(",")]
    lasso_alphas  = [float(x) for x in args.lasso_alphas.split(",")]
    pls_ncomp_grid = [int(x) for x in args.pls_ncomp_grid.split(",")]

    # ------------------------------------------------------------------
    # NDM pre-computation (done once; per-fold fitting inside loop)
    # ------------------------------------------------------------------
    ndm_ready        = False
    ndm_evals_all    = None
    ndm_evecs_all    = None
    ndm_x0_list      = []   # list of per-seed x0 vectors
    ndm_seed_groups  = []   # list of per-seed ROI id lists (1-based)
    ndm_beta_grid    = None
    ndm_time_grid    = None
    use_node_keys_base = list(use_node_keys)  # saved for NDM augmentation

    if args.ndm_enable:
        if args.ndm_edge_key not in edge_names:
            raise RuntimeError(f"--ndm-edge-key '{args.ndm_edge_key}' not in edge_feature_names: {edge_names}")
        ndm_c   = edge_names.index(args.ndm_edge_key)
        W_all   = A[:, ndm_c].astype(np.float32)
        W_all   = np.clip(W_all, 0.0, None)
        if args.ndm_log1p_w:
            W_all = np.log1p(W_all)

        # Parse seeds: support two modes
        #   (a) grouped  "115,116;213,214;37,38"  → 3 separate seed groups, each gets own NDM
        #   (b) flat     "115,116,213,214"         → treated as ONE combined seed group
        # Grouped mode allows multi-seed ablation; flat mode is legacy single-group behavior.
        raw_seeds = args.ndm_seed_ids
        if ";" in raw_seeds:
            # Grouped: each semicolon-separated token is one seed group
            groups = [g.strip() for g in raw_seeds.split(";") if g.strip()]
            ndm_x0_list = []
            ndm_seed_groups = []
            for g in groups:
                ids1 = parse_int_list_arg(g)
                ids0 = [s - 1 for s in ids1 if 1 <= s <= n_nodes]
                if not ids0:
                    continue
                x0 = np.zeros(n_nodes, dtype=np.float32)
                x0[ids0] = 1.0
                x0 /= x0.sum()
                ndm_x0_list.append(x0)
                ndm_seed_groups.append(ids1)
        else:
            # Flat: single group (legacy)
            ids1 = parse_int_list_arg(raw_seeds)
            ids0 = [s - 1 for s in ids1 if 1 <= s <= n_nodes]
            if not ids0:
                raise RuntimeError("No valid --ndm-seed-ids")
            x0 = np.zeros(n_nodes, dtype=np.float32)
            x0[ids0] = 1.0
            x0 /= x0.sum()
            ndm_x0_list = [x0]
            ndm_seed_groups = [ids1]

        if not ndm_x0_list:
            raise RuntimeError("No valid NDM seed groups parsed from --ndm-seed-ids")

        ndm_beta_grid = parse_float_list_arg(args.ndm_beta_grid)
        ndm_time_grid = parse_float_list_arg(args.ndm_time_grid)

        print("Pre-computing NDM eigendecompositions …")
        ndm_evals_all, ndm_evecs_all = ndm_precompute_eig(W_all)
        ndm_ready = True
        print(f"  NDM seed groups ({len(ndm_x0_list)} total): {ndm_seed_groups}")
        print(f"  NDM output dims: {len(ndm_x0_list)} × {n_nodes} = {len(ndm_x0_list)*n_nodes}")
        print(f"  β grid: {ndm_beta_grid}")
        print(f"  t grid: {ndm_time_grid}\n")

    # Diagnostic group counts
    for label, code in LABEL_MAP.items():
        print(f"  {label}: {np.sum(dx_code_all == code)} subjects")
    print(f"Subjects={n_subj}  SC_feat={n_sc_feat}  ROIs={n_nodes}  Edges={n_edges}\n")

    subj_splits = build_subject_kfold(n_subj, args.cv_folds, args.seed)

    # ------------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------------
    # OOF pair predictions (upper-triangle pairs from test subjects)
    oof_pairs_true:  Dict[Tuple[int,int], np.ndarray] = {}
    oof_pairs_pred:  Dict[Tuple[int,int], np.ndarray] = {}
    oof_pairs_dx:    Dict[Tuple[int,int], Tuple[int,int]] = {}

    # Weighted edge importance aggregation
    imp_overall_weighted = np.zeros(n_edges, dtype=np.float64)
    imp_pairtype_weighted: Dict[Tuple[int,int], np.ndarray] = {
        k: np.zeros(n_edges, dtype=np.float64) for k in PAIR_TYPE_NAMES
    }
    imp_weight_overall = 0.0
    imp_weight_pairtype: Dict[Tuple[int,int], float] = {k: 0.0 for k in PAIR_TYPE_NAMES}

    # Stability tracking
    topk = args.topk_edges
    topk_sets_overall:  List[set] = []
    topk_sets_pairtype: Dict[Tuple[int,int], List[set]] = {k: [] for k in PAIR_TYPE_NAMES}

    fold_rows = []

    # ------------------------------------------------------------------
    # k-fold CV
    # ------------------------------------------------------------------
    for fold_i, (train_idx, test_idx) in enumerate(subj_splits, start=1):
        tag = f"[fold {fold_i}/{args.cv_folds}]"

        rng_f    = np.random.default_rng(args.seed + fold_i * 997)
        n_val    = max(1, int(len(train_idx) * args.val_ratio))
        val_idx  = rng_f.choice(train_idx, size=n_val, replace=False)
        fit_idx  = np.setdiff1d(train_idx, val_idx)

        # Normalise SC features on fit-train (SC only — NDM kept separate below)
        mu = feat_all[fit_idx].mean(axis=0, keepdims=True)
        sd = feat_all[fit_idx].std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-6, 1.0, sd)
        feat_n = ((feat_all - mu) / sd).astype(np.float32)

        # ---- NDM: fit per-fold, produce per-subject diffusion predictions ----
        # NDM features are kept SEPARATE from PCA to preserve their physical signal.
        # PCA is applied to SC features only; NDM differences are concatenated after.
        ndm_fold_info    = {"enabled": False}
        ndm_n            = None   # (n_subj, n_seeds * n_nodes) normalised NDM predictions, or None
        ndm_tau_baseline = None   # (n_subj, n_nodes) raw tau-scale baseline for residual mode
        if ndm_ready:
            evals_fit = [ndm_evals_all[i] for i in fit_idx]
            evecs_fit = [ndm_evecs_all[i] for i in fit_idx]

            # Fit one NDM per seed group; concatenate outputs
            ndm_parts     = []   # normalised, for concatenation as features
            ndm_raw_parts = []   # raw (tau SUVR scale), for residual baseline
            seed_infos    = []
            for k, x0_k in enumerate(ndm_x0_list):
                best_k = fit_ndm_global_params(
                    evals_train=evals_fit,
                    evecs_train=evecs_fit,
                    y_train=y[fit_idx],
                    x0=x0_k,
                    beta_grid=ndm_beta_grid,
                    time_grid=ndm_time_grid,
                )
                ndm_k = ndm_predict_from_eig(
                    ndm_evals_all, ndm_evecs_all,
                    x0=x0_k, gamma=best_k["gamma"],
                ).astype(np.float32)  # (n_subj, n_nodes)

                # Keep raw predictions for residual baseline (tau SUVR scale)
                ndm_raw_parts.append(ndm_k.copy())

                # Normalise per seed group on fit-train only (for feature concatenation mode)
                mu_k = ndm_k[fit_idx].mean(axis=0, keepdims=True)
                sd_k = ndm_k[fit_idx].std(axis=0, keepdims=True)
                sd_k = np.where(sd_k < 1e-6, 1.0, sd_k)
                ndm_parts.append((ndm_k - mu_k) / sd_k)

                seed_infos.append({
                    "seed_group": ndm_seed_groups[k],
                    "beta": float(best_k["beta"]),
                    "t":    float(best_k["t"]),
                    "gamma": float(best_k["gamma"]),
                    "mse":  float(best_k["mse"]),
                })
                print(f"{tag} NDM seed{k+1}={ndm_seed_groups[k]}  "
                      f"β={best_k['beta']:.3f}  t={best_k['t']:.2f}  mse={best_k['mse']:.4f}")

            # Concatenate all seed outputs: (n_subj, n_seeds * n_nodes)
            ndm_n = np.concatenate(ndm_parts, axis=1).astype(np.float32)

            # For residual mode: pick the seed with the best (lowest) train MSE
            # as the physics baseline for τ_NDM.  Raw (un-normalised) predictions
            # are already in tau SUVR scale due to the gamma scaling inside ndm_predict_from_eig.
            best_seed_idx = int(np.argmin([s["mse"] for s in seed_infos]))
            ndm_tau_baseline = ndm_raw_parts[best_seed_idx]   # (n_subj, n_nodes)

            ndm_fold_info = {
                "enabled":          True,
                "n_seeds":          len(ndm_x0_list),
                "seed_infos":       seed_infos,
                "total_ndm_dims":   ndm_n.shape[1],
                "best_seed_idx":    best_seed_idx,
                "baseline_seed":    ndm_seed_groups[best_seed_idx],
            }

        # PCA on SC features only (full rank of fit-train)
        # When --ndm-only: skip SC-PCA entirely; ΔNDM is the sole input.
        n_feat_now = feat_n.shape[1]
        if args.ndm_only:
            # Dummy PCA with 0 components — coef_to_edge_importance will return zeros,
            # which is correct: SC edges have no attribution in NDM-only ablation.
            pca      = PCA(n_components=1, random_state=args.seed)
            pca.fit(feat_n[fit_idx])          # fit for API consistency
            feat_pca = np.zeros((n_subj, 0), dtype=np.float32)  # empty
            n_comp   = 0
            var_exp  = 0.0
            print(f"{tag} NDM-only mode: SC-PCA skipped  "
                  f"NDM: {ndm_n.shape[1]} dims")
        else:
            n_comp = min(
                args.pca_components if args.pca_components > 0 else 9999,
                len(fit_idx) - 1,
                n_feat_now,
            )
            pca = PCA(n_components=n_comp, random_state=args.seed)
            pca.fit(feat_n[fit_idx])
            feat_pca = pca.transform(feat_n).astype(np.float32)  # (n_subj, n_comp)
            var_exp  = float(np.sum(pca.explained_variance_ratio_))

            if ndm_n is not None:
                print(f"{tag} PCA(SC): {n_feat_now} → {n_comp} dims  ({var_exp*100:.1f}% var)"
                      f"  + NDM: {ndm_n.shape[1]} dims  →  total {n_comp + ndm_n.shape[1]} dims")
            else:
                print(f"{tag} PCA: {n_feat_now} → {n_comp} dims  ({var_exp*100:.1f}% var)")

        ndm_baseline_mse_fit = float("nan")  # set below if residual mode

        # Build pairwise datasets
        # Residual mode: NDM predicts physics baseline; Ridge learns Δ(tau - tau_NDM) from ΔSC.
        # Feature concat mode: X = [ΔZ_sc || ΔNDM]; Ridge learns Δtau from both.
        if args.ndm_residual and ndm_n is not None:
            # Compute per-subject residual = tau_observed - tau_NDM_best_seed
            # ndm_tau_baseline is in tau SUVR scale (gamma-scaled by ndm_predict_from_eig)
            y_target = (y - ndm_tau_baseline).astype(np.float32)
            ndm_baseline_mse_fit = float(np.mean((y[fit_idx] - ndm_tau_baseline[fit_idx])**2))
            print(f"{tag} Residual mode: NDM baseline MSE on fit-train = {ndm_baseline_mse_fit:.4f}  "
                  f"(seed={ndm_fold_info['baseline_seed']})")
        else:
            y_target = y  # standard mode: predict raw tau differences

        X_tr, y_tr, gi_tr, gj_tr, dx_tr = build_pairs(feat_pca, y_target, dx_code_all, fit_idx, augment=augment)
        X_vl, y_vl, gi_vl, gj_vl, _     = build_pairs(feat_pca, y_target, dx_code_all, val_idx,  augment=False)
        X_te, y_te, gi_te, gj_te, dx_te  = build_pairs(feat_pca, y_target, dx_code_all, test_idx, augment=False)

        if ndm_n is not None and not args.ndm_residual:
            # Feature concatenation mode: append ΔNDM to ΔSC-PCA
            ndm_tr = (ndm_n[gi_tr] - ndm_n[gj_tr]).astype(np.float32)
            ndm_vl = (ndm_n[gi_vl] - ndm_n[gj_vl]).astype(np.float32)
            ndm_te = (ndm_n[gi_te] - ndm_n[gj_te]).astype(np.float32)
            X_tr = np.concatenate([X_tr, ndm_tr], axis=1)
            X_vl = np.concatenate([X_vl, ndm_vl], axis=1)
            X_te = np.concatenate([X_te, ndm_te], axis=1)

        ndm_dim = ndm_n.shape[1] if ndm_n is not None else 0
        print(f"{tag} pairs  train={len(X_tr):,}  val={len(X_vl):,}  test={len(X_te):,}"
              f"  input_dim={n_comp}+{ndm_dim}={n_comp+ndm_dim}")

        # ---- Overall Ridge model ----
        model_overall, best_alpha, val_score = fit_ridge_select_alpha(
            X_tr, y_tr, X_vl, y_vl, ridge_alphas
        )
        pred_te  = model_overall.predict(X_te).astype(np.float32)
        pred_zero = np.zeros_like(y_te)
        perf_te  = evaluate_pairs(y_te, pred_te)
        perf_zero = evaluate_pairs(y_te, pred_zero)

        print(f"{tag} Overall  alpha={best_alpha:.1f}  val_r={val_score:.4f}  "
              f"test_pearson={perf_te['pearson_by_pair']:.4f}  r2={perf_te['r2_flat']:.4f}")

        # ---- Edge importance: overall ----
        imp_overall = coef_to_edge_importance(model_overall, pca, n_edges, n_nodes, use_edge_keys)
        w_fold = float(len(X_te))
        imp_overall_weighted += imp_overall * w_fold
        imp_weight_overall   += w_fold
        topk_sets_overall.append(set(np.argsort(imp_overall)[-topk:].tolist()))

        # OOF storage
        for k in range(len(gi_te)):
            key = (int(gi_te[k]), int(gj_te[k]))
            oof_pairs_true[key] = y_te[k]
            oof_pairs_pred[key] = pred_te[k]
            oof_pairs_dx[key]   = (int(dx_te[k, 0]), int(dx_te[k, 1]))

        # ---- Optional: PLS and Lasso (only if --compare-models) ----
        if args.compare_models:
            model_pls, best_ncomp, val_pls = fit_pls_select_ncomp(
                X_tr, y_tr, X_vl, y_vl, pls_ncomp_grid
            )
            pred_pls = model_pls.predict(X_te).astype(np.float32)
            perf_pls = evaluate_pairs(y_te, pred_pls)
            print(f"{tag} PLS     n_comp={best_ncomp}  val_r={val_pls:.4f}  "
                  f"test_pearson={perf_pls['pearson_by_pair']:.4f}  r2={perf_pls['r2_flat']:.4f}")

            model_lasso, best_lasso_alpha, val_lasso = fit_lasso_select_alpha(
                X_tr, y_tr, X_vl, y_vl, lasso_alphas
            )
            pred_lasso = model_lasso.predict(X_te).astype(np.float32)
            perf_lasso = evaluate_pairs(y_te, pred_lasso)
            print(f"{tag} Lasso   alpha={best_lasso_alpha:.1f}  val_r={val_lasso:.4f}  "
                  f"test_pearson={perf_lasso['pearson_by_pair']:.4f}  r2={perf_lasso['r2_flat']:.4f}")
        else:
            best_ncomp, val_pls, perf_pls = 0, float("nan"), {"pearson_by_pair": float("nan"), "r2_flat": float("nan")}
            best_lasso_alpha, val_lasso, perf_lasso = float("nan"), float("nan"), {"pearson_by_pair": float("nan"), "r2_flat": float("nan")}

        # ---- Pair-type stratified Ridge ----
        pt_results = {}
        for pt_key, pt_name in PAIR_TYPE_NAMES.items():
            dx_lo, dx_hi = pt_key
            # Training pairs of this type
            tr_mask = (dx_tr[:, 0] == dx_lo) & (dx_tr[:, 1] == dx_hi)
            te_mask = (dx_te[:, 0] == dx_lo) & (dx_te[:, 1] == dx_hi)
            n_tr_pt = int(np.sum(tr_mask))
            n_te_pt = int(np.sum(te_mask))

            if n_tr_pt < MIN_PAIRS_FOR_STRATIFIED or n_te_pt == 0:
                print(f"{tag}   {pt_name:10s} skip (train={n_tr_pt}, test={n_te_pt})")
                pt_results[pt_name] = None
                continue

            m_pt, alpha_pt, _ = fit_ridge_select_alpha(
                X_tr[tr_mask], y_tr[tr_mask],
                X_vl, y_vl,    # use same val set for consistency
                ridge_alphas,
            )
            pred_pt  = m_pt.predict(X_te[te_mask]).astype(np.float32)
            perf_pt  = evaluate_pairs(y_te[te_mask], pred_pt)
            perf_z_pt = evaluate_pairs(y_te[te_mask], np.zeros_like(pred_pt))

            imp_pt = coef_to_edge_importance(m_pt, pca, n_edges, n_nodes, use_edge_keys)
            w_pt   = float(n_te_pt)
            imp_pairtype_weighted[pt_key] += imp_pt * w_pt
            imp_weight_pairtype[pt_key]   += w_pt
            topk_sets_pairtype[pt_key].append(set(np.argsort(imp_pt)[-topk:].tolist()))

            pt_results[pt_name] = {
                "n_train_pairs": n_tr_pt, "n_test_pairs": n_te_pt,
                "alpha": alpha_pt,
                "pearson": perf_pt["pearson_by_pair"],
                "pearson_zero": perf_z_pt["pearson_by_pair"],
                "r2": perf_pt["r2_flat"],
            }
            print(f"{tag}   {pt_name:10s}  n_tr={n_tr_pt:5d}  "
                  f"pearson={perf_pt['pearson_by_pair']:.4f}  r2={perf_pt['r2_flat']:.4f}")

        fold_rows.append({
            "fold": fold_i,
            "n_fit": len(fit_idx), "n_val": len(val_idx), "n_test": len(test_idx),
            "ndm_enabled":       ndm_fold_info["enabled"],
            "ndm_n_seeds":       ndm_fold_info.get("n_seeds", 0),
            "ndm_total_dims":    ndm_fold_info.get("total_ndm_dims", 0),
            "ndm_mean_mse":      float(np.mean([s["mse"] for s in ndm_fold_info["seed_infos"]])) if ndm_fold_info["enabled"] else float("nan"),
            "ndm_residual_mode": args.ndm_residual,
            "ndm_baseline_seed": ndm_fold_info.get("baseline_seed", []),
            "ndm_baseline_mse":  ndm_baseline_mse_fit if args.ndm_residual and ndm_fold_info["enabled"] else float("nan"),
            "pca_components": n_comp, "pca_var_explained": var_exp,
            "ridge_alpha": best_alpha,
            "ridge_val_pearson":  val_score,
            "ridge_test_pearson": perf_te["pearson_by_pair"],
            "ridge_test_r2":      perf_te["r2_flat"],
            "pls_ncomp":          best_ncomp,
            "pls_val_pearson":    val_pls,
            "pls_test_pearson":   perf_pls["pearson_by_pair"],
            "pls_test_r2":        perf_pls["r2_flat"],
            "lasso_alpha":        best_lasso_alpha,
            "lasso_val_pearson":  val_lasso,
            "lasso_test_pearson": perf_lasso["pearson_by_pair"],
            "lasso_test_r2":      perf_lasso["r2_flat"],
            "zero_test_pearson":  perf_zero["pearson_by_pair"],
            # keep backward-compat alias
            "overall_alpha": best_alpha,
            "overall_val_pearson": val_score,
            "overall_test_pearson": perf_te["pearson_by_pair"],
            "overall_test_r2": perf_te["r2_flat"],
            **{f"{name.replace('-','_')}_test_pearson": (
                pt_results[name]["pearson"] if pt_results.get(name) else float("nan"))
               for name in PAIR_TYPE_NAMES.values()},
        })
        print()

    # ------------------------------------------------------------------
    # OOF aggregated metrics
    # ------------------------------------------------------------------
    keys      = sorted(oof_pairs_true.keys())
    y_oof     = np.stack([oof_pairs_true[k] for k in keys])
    pred_oof  = np.stack([oof_pairs_pred[k] for k in keys])
    dx_oof    = np.array([oof_pairs_dx[k] for k in keys], dtype=np.int32)

    perf_oof  = evaluate_pairs(y_oof, pred_oof)
    perf_zero_oof = evaluate_pairs(y_oof, np.zeros_like(y_oof))
    perm_overall = None
    perm_by_pair_type: Dict[str, Dict[str, float]] = {}
    perm_rng = None
    perm_seed = None
    if args.perm_test and args.n_perm > 0:
        perm_seed = args.seed + 2026 if args.perm_seed is None else args.perm_seed
        perm_rng = np.random.default_rng(perm_seed)
        perm_overall = permutation_test_mean_pearson_by_pair(
            y_oof, pred_oof, args.n_perm, perm_rng
        )

    print("=== OOF OVERALL ===")
    print(f"  Zero   pearson={perf_zero_oof['pearson_by_pair']}  r2={perf_zero_oof['r2_flat']:.4f}")
    print(f"  Ridge  pearson={perf_oof['pearson_by_pair']:.4f}  r2={perf_oof['r2_flat']:.4f}")
    if perm_overall is not None:
        print(f"  Perm   p(one-sided, r>random)={perm_overall['p_value_gt_random']:.6g}  "
              f"null_mean={perm_overall['null_mean']:.4f}  n_perm={perm_overall['n_perm_valid']}")

    # OOF per pair type
    print("\n=== OOF PER PAIR TYPE ===")
    pt_oof_rows = []
    for pt_key, pt_name in PAIR_TYPE_NAMES.items():
        dx_lo, dx_hi = pt_key
        mask = (dx_oof[:, 0] == dx_lo) & (dx_oof[:, 1] == dx_hi)
        if not np.any(mask):
            continue
        pm = evaluate_pairs(y_oof[mask], pred_oof[mask])
        pz = evaluate_pairs(y_oof[mask], np.zeros_like(y_oof[mask]))
        pm_perm = None
        if perm_rng is not None and args.perm_test_pair_types:
            pm_perm = permutation_test_mean_pearson_by_pair(
                y_oof[mask], pred_oof[mask], args.n_perm, perm_rng
            )
            perm_by_pair_type[pt_name] = pm_perm
        print(f"  {pt_name:10s}  n={np.sum(mask):5d}  "
              f"pearson_zero={pz['pearson_by_pair']}  "
              f"pearson_ridge={pm['pearson_by_pair']:.4f}  r2={pm['r2_flat']:.4f}")
        pt_oof_rows.append({
            "pair_type": pt_name, "n_pairs": int(np.sum(mask)),
            "pearson_zero":  pz["pearson_by_pair"],
            "r2_zero":       pz["r2_flat"],
            "pearson_ridge": pm["pearson_by_pair"],
            "r2_ridge":      pm["r2_flat"],
            "perm_p_ridge_gt_random": (
                pm_perm["p_value_gt_random"] if pm_perm is not None else float("nan")
            ),
        })

    # ------------------------------------------------------------------
    # Aggregate edge importance
    # ------------------------------------------------------------------
    imp_overall_agg = imp_overall_weighted / (imp_weight_overall + 1e-12)

    # Per pair type
    imp_pt_agg: Dict[Tuple[int,int], Optional[np.ndarray]] = {}
    for pt_key in PAIR_TYPE_NAMES:
        w = imp_weight_pairtype[pt_key]
        imp_pt_agg[pt_key] = (
            imp_pairtype_weighted[pt_key] / w if w > 0 else None
        )

    # Hub scores
    hub_overall = compute_hub_scores(imp_overall_agg, n_nodes, tri)
    hub_pt: Dict[Tuple[int,int], Optional[np.ndarray]] = {
        k: compute_hub_scores(v, n_nodes, tri) if v is not None else None
        for k, v in imp_pt_agg.items()
    }

    # Stability
    stab_overall = mean_jaccard(topk_sets_overall)
    stab_pt = {
        PAIR_TYPE_NAMES[k]: mean_jaccard(topk_sets_pairtype[k])
        for k in PAIR_TYPE_NAMES if len(topk_sets_pairtype[k]) >= 2
    }
    print(f"\n=== TOP-{topk} EDGE STABILITY (Jaccard) ===")
    print(f"  Overall: {stab_overall:.4f}")
    for name, j in stab_pt.items():
        print(f"  {name:10s}: {j:.4f}")

    mean_tau = np.nanmean(y_oof, axis=0)  # (n_nodes,)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    # 1. Fold metrics
    write_csv(os.path.join(args.out_dir, "pairwise_attr_fold_metrics.csv"), fold_rows)

    # 2. OOF pair-level predictions
    pair_i_idx = np.array([k[0] for k in keys], dtype=np.int32)
    pair_j_idx = np.array([k[1] for k in keys], dtype=np.int32)
    pair_sid_i = subject_ids[pair_i_idx]
    pair_sid_j = subject_ids[pair_j_idx]
    y_oof_delta_tau = (y[pair_i_idx] - y[pair_j_idx]).astype(np.float32)
    np.savez_compressed(
        os.path.join(args.out_dir, "pairwise_attr_oof_predictions.npz"),
        pair_i_idx=pair_i_idx,
        pair_j_idx=pair_j_idx,
        pair_subject_id_i=pair_sid_i,
        pair_subject_id_j=pair_sid_j,
        pair_dx=dx_oof.astype(np.int32),
        y_true=y_oof.astype(np.float32),      # training/eval target (can be residual delta tau)
        y_pred=pred_oof.astype(np.float32),
        y_true_delta_tau=y_oof_delta_tau,     # always raw delta tau
        target_name=np.array(
            ["delta_tau_residual" if args.ndm_residual else "delta_tau"],
            dtype=np.str_,
        ),
        residual_mode=np.array([1 if args.ndm_residual else 0], dtype=np.int8),
    )

    # 3. Overall edge importance (all edges)
    edge_rows = []
    for k in range(n_edges):
        row = {
            "edge_idx":     k,
            "roi_i_1based": int(tri[0][k] + 1),
            "roi_j_1based": int(tri[1][k] + 1),
            "imp_overall":  float(imp_overall_agg[k]),
        }
        for pt_key, pt_name in PAIR_TYPE_NAMES.items():
            pt_tag = pt_name.replace("-", "_")
            row[f"imp_{pt_tag}"] = (
                float(imp_pt_agg[pt_key][k]) if imp_pt_agg[pt_key] is not None else float("nan")
            )
        edge_rows.append(row)
    edge_rows.sort(key=lambda x: x["imp_overall"], reverse=True)
    write_csv(os.path.join(args.out_dir, "pairwise_attr_edge_importance.csv"), edge_rows)

    # 3. Top-K edges
    write_csv(os.path.join(args.out_dir, "pairwise_attr_edge_topk.csv"), edge_rows[:topk])

    # 4. Hub ROI scores
    hub_rows = []
    for r in range(n_nodes):
        row = {
            "roi_1based":       r + 1,
            "hub_score_overall": float(hub_overall[r]),
            "mean_true_tau":    float(mean_tau[r]),
        }
        for pt_key, pt_name in PAIR_TYPE_NAMES.items():
            pt_tag = pt_name.replace("-", "_")
            row[f"hub_{pt_tag}"] = (
                float(hub_pt[pt_key][r]) if hub_pt[pt_key] is not None else float("nan")
            )
        hub_rows.append(row)
    hub_rows.sort(key=lambda x: x["hub_score_overall"], reverse=True)
    write_csv(os.path.join(args.out_dir, "pairwise_attr_hub_roi_scores.csv"), hub_rows)

    # 5. OOF pair type metrics
    write_csv(os.path.join(args.out_dir, "pairwise_attr_pair_type_metrics.csv"), pt_oof_rows)

    # 6. Per pair-type top-K edges (separate CSVs)
    for pt_key, pt_name in PAIR_TYPE_NAMES.items():
        if imp_pt_agg[pt_key] is None:
            continue
        pt_tag   = pt_name.replace("-", "_")
        pt_edges = sorted([
            {"edge_idx": k,
             "roi_i_1based": int(tri[0][k] + 1),
             "roi_j_1based": int(tri[1][k] + 1),
             "imp": float(imp_pt_agg[pt_key][k])}
            for k in range(n_edges)
        ], key=lambda x: -x["imp"])
        write_csv(
            os.path.join(args.out_dir, f"pairwise_attr_edge_topk_{pt_tag}.csv"),
            pt_edges[:topk],
        )

    # 7. Per pair-type hub ROIs (separate CSVs)
    for pt_key, pt_name in PAIR_TYPE_NAMES.items():
        if hub_pt[pt_key] is None:
            continue
        pt_tag   = pt_name.replace("-", "_")
        pt_hubs  = sorted([
            {"roi_1based": r + 1,
             "hub_score":  float(hub_pt[pt_key][r]),
             "mean_true_tau": float(mean_tau[r])}
            for r in range(n_nodes)
        ], key=lambda x: -x["hub_score"])
        write_csv(
            os.path.join(args.out_dir, f"pairwise_attr_hub_roi_{pt_tag}.csv"),
            pt_hubs,
        )

    # 8. Stability JSON
    with open(os.path.join(args.out_dir, "pairwise_attr_stability.json"), "w") as f:
        json.dump({
            "topk": topk,
            "overall_jaccard":     stab_overall,
            "pairtype_jaccard":    stab_pt,
        }, f, indent=2)

    # 9. Meta JSON
    def fmean(key):
        vals = [r.get(key, float("nan")) for r in fold_rows]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    meta = {
        "n_subjects": n_subj, "n_nodes": n_nodes,
        "n_sc_features": n_sc_feat, "n_edges": n_edges,
        "mode": "ndm_only" if args.ndm_only else ("sc_ndm" if ndm_ready else "sc_only"),
        "ndm": {
            "enabled":           ndm_ready,
            "residual_mode":     args.ndm_residual,
            "n_seeds":           len(ndm_x0_list) if ndm_ready else 0,
            "seed_groups":       ndm_seed_groups if ndm_ready else [],
            "edge_key":          args.ndm_edge_key if ndm_ready else None,
            "log1p_w":           args.ndm_log1p_w if ndm_ready else None,
            "mean_mse_per_fold": [r["ndm_mean_mse"] for r in fold_rows],
            "baseline_mse_per_fold": [r["ndm_baseline_mse"] for r in fold_rows],
        },
        "pca_components_used": int(fold_rows[0]["pca_components"]),
        "pca_var_explained_mean": float(np.mean([r["pca_var_explained"] for r in fold_rows])),
        "cv_folds": args.cv_folds,
        "augment_training_pairs": augment,
        "oof_overall": {**perf_oof, "zero_pearson": perf_zero_oof["pearson_by_pair"]},
        "oof_by_pair_type": {r["pair_type"]: r for r in pt_oof_rows},
        "permutation_test": {
            "enabled": bool(args.perm_test and args.n_perm > 0),
            "n_perm": int(args.n_perm),
            "seed": perm_seed,
            "overall": perm_overall,
            "by_pair_type": perm_by_pair_type if args.perm_test_pair_types else {},
        },
        "fold_summary": {
            "ridge_test_pearson_mean": fmean("ridge_test_pearson"),
            "ridge_test_pearson_std":  float(np.std([r["ridge_test_pearson"] for r in fold_rows], ddof=1)),
            "ridge_test_r2_mean":      fmean("ridge_test_r2"),
            "pls_test_pearson_mean":   fmean("pls_test_pearson"),
            "pls_test_pearson_std":    float(np.std([r["pls_test_pearson"] for r in fold_rows], ddof=1)),
            "pls_test_r2_mean":        fmean("pls_test_r2"),
            "lasso_test_pearson_mean": fmean("lasso_test_pearson"),
            "lasso_test_pearson_std":  float(np.std([r["lasso_test_pearson"] for r in fold_rows], ddof=1)),
            "lasso_test_r2_mean":      fmean("lasso_test_r2"),
            # backward compat
            "overall_test_pearson_mean": fmean("overall_test_pearson"),
            "overall_test_pearson_std":  float(np.std([r["overall_test_pearson"] for r in fold_rows], ddof=1)),
            "overall_test_r2_mean":      fmean("overall_test_r2"),
            **{f"{name.replace('-','_')}_pearson_mean": fmean(f"{name.replace('-','_')}_test_pearson")
               for name in PAIR_TYPE_NAMES.values()},
        },
        "top_k_edge_stability": {
            "k": topk,
            "overall_jaccard":  stab_overall,
            "pairtype_jaccard": stab_pt,
        },
        "ridge_alpha_per_fold": [r["overall_alpha"] for r in fold_rows],
    }
    with open(os.path.join(args.out_dir, "pairwise_attr_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------
    # Final summary table
    # ------------------------------------------------------------------
    target_label = "Δ(tau - tau_NDM)" if args.ndm_residual else "Δtau"
    print("\n=== FINAL SUMMARY ===")
    print(f"  Target: {target_label}")
    if args.ndm_residual:
        ndm_mean_baseline_mse = float(np.nanmean([r["ndm_baseline_mse"] for r in fold_rows]))
        print(f"  NDM baseline MSE (fit-train mean): {ndm_mean_baseline_mse:.4f}")
    print(f"  {'Model':<10}  {'Pearson':>8}  {'Note'}")
    print(f"  {'Ridge':<10}  {perf_oof['pearson_by_pair']:>8.4f}  (OOF)")
    if perm_overall is not None:
        print(f"  {'Perm p':<10}  {perm_overall['p_value_gt_random']:>8.4g}  (H1: r > random)")
    if args.compare_models:
        pls_pearson_mean   = float(np.nanmean([r["pls_test_pearson"]   for r in fold_rows]))
        lasso_pearson_mean = float(np.nanmean([r["lasso_test_pearson"] for r in fold_rows]))
        print(f"  {'PLS':<10}  {pls_pearson_mean:>8.4f}  (fold mean)")
        print(f"  {'Lasso':<10}  {lasso_pearson_mean:>8.4f}  (fold mean)")
    print(f"  Top-{topk} edge stability (Jaccard, Ridge) = {stab_overall:.4f}")
    print(f"\n  Per pair-type OOF Pearson:")
    for r in pt_oof_rows:
        print(f"    {r['pair_type']:10s}  n={r['n_pairs']:5d}  "
              f"zero={str(r['pearson_zero']):>6}  ridge={r['pearson_ridge']:.4f}")
    print(f"\n  Per pair-type edge stability (Jaccard):")
    for name, j in stab_pt.items():
        print(f"    {name:10s}  {j:.4f}")
    print(f"\n  OOF pair-level predictions → {os.path.join(args.out_dir, 'pairwise_attr_oof_predictions.npz')}")
    print(f"\nOutputs → {args.out_dir}")


if __name__ == "__main__":
    main()

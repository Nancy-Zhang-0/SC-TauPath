#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def parse_list_arg(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_float_list_arg(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list_arg(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(n: int, train_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(round(n * train_ratio))
    train_idx = np.sort(idx[:n_train])
    test_idx = np.sort(idx[n_train:])
    return train_idx, test_idx


def split_train_val(train_idx: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if val_ratio <= 0.0:
        return train_idx.copy(), np.array([], dtype=np.int32)
    n = len(train_idx)
    n_val = max(1, int(round(n * val_ratio)))
    rng = np.random.default_rng(seed + 1)
    perm = train_idx.copy()
    rng.shuffle(perm)
    val_idx = np.sort(perm[:n_val])
    tr_idx = np.sort(perm[n_val:])
    if tr_idx.size == 0:
        raise RuntimeError("Validation split leaves no training samples. Reduce --val-ratio.")
    return tr_idx, val_idx


def standardize_train_test(X_train: np.ndarray, X_other: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (X_train - mu) / sd, (X_other - mu) / sd


def build_full_network_features(
    A: np.ndarray,
    X: np.ndarray,
    edge_names: List[str],
    node_names: List[str],
    use_edge_keys: List[str],
    use_node_keys: List[str],
) -> np.ndarray:
    # A: [B, C_edge, N, N], X: [B, N, C_node]
    b, _, n, _ = A.shape
    tri = np.triu_indices(n, k=1)
    edge_idx = [edge_names.index(k) for k in use_edge_keys]
    node_idx = [node_names.index(k) for k in use_node_keys]

    rows = []
    for i in range(b):
        vecs = []
        for c in edge_idx:
            vecs.append(A[i, c][tri].astype(np.float32))
        for c in node_idx:
            vecs.append(X[i, :, c].astype(np.float32))
        if len(vecs) == 0:
            raise RuntimeError("No features selected. Check --edge-keys/--node-keys")
        rows.append(np.concatenate(vecs, axis=0))
    return np.stack(rows, axis=0).astype(np.float32)


def build_normalized_laplacian(w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    w = 0.5 * (w + w.T)
    np.fill_diagonal(w, 0.0)
    d = np.sum(w, axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(d + eps)
    return np.eye(w.shape[0], dtype=np.float64) - (d_inv_sqrt[:, None] * w * d_inv_sqrt[None, :])


def ndm_precompute_eig(w_batch: np.ndarray):
    evals = []
    evecs = []
    for i in range(w_batch.shape[0]):
        lam, u = np.linalg.eigh(build_normalized_laplacian(w_batch[i]))
        evals.append(lam.astype(np.float64))
        evecs.append(u.astype(np.float64))
    return evals, evecs


def ndm_predict_from_eig(evals, evecs, x0: np.ndarray, gamma: float) -> np.ndarray:
    out = []
    x0 = x0.astype(np.float64)
    for lam, u in zip(evals, evecs):
        coeff = u.T @ x0
        x = u @ (np.exp(-gamma * lam) * coeff)
        out.append(x.astype(np.float32))
    return np.stack(out, axis=0)


def fit_ndm_global_params(evals_train, evecs_train, y_train: np.ndarray, x0: np.ndarray, beta_grid, time_grid):
    best = {"mse": float("inf"), "beta": None, "t": None, "gamma": None}
    for beta in beta_grid:
        for t in time_grid:
            gamma = float(beta) * float(t)
            pred = ndm_predict_from_eig(evals_train, evecs_train, x0=x0, gamma=gamma)
            mse = float(np.mean((pred - y_train) ** 2))
            if mse < best["mse"]:
                best = {"mse": mse, "beta": float(beta), "t": float(t), "gamma": gamma}
    return best


def fit_ndm_subject_params(evals_train, evecs_train, y_train: np.ndarray, x0: np.ndarray, beta_grid, time_grid):
    n = len(evals_train)
    best_beta = np.zeros((n,), dtype=np.float64)
    best_t = np.zeros((n,), dtype=np.float64)
    best_gamma = np.zeros((n,), dtype=np.float64)
    best_mse = np.full((n,), np.inf, dtype=np.float64)

    for i in range(n):
        yi = y_train[i : i + 1]
        for beta in beta_grid:
            for t in time_grid:
                gamma = float(beta) * float(t)
                pred_i = ndm_predict_from_eig([evals_train[i]], [evecs_train[i]], x0=x0, gamma=gamma)
                mse_i = float(np.mean((pred_i - yi) ** 2))
                if mse_i < best_mse[i]:
                    best_mse[i] = mse_i
                    best_beta[i] = float(beta)
                    best_t[i] = float(t)
                    best_gamma[i] = gamma
    return {
        "beta": best_beta,
        "t": best_t,
        "gamma": best_gamma,
        "mse": best_mse,
    }


def ndm_predict_from_eig_subject_gammas(evals, evecs, x0: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    out = []
    x0 = x0.astype(np.float64)
    for lam, u, gamma in zip(evals, evecs, gammas):
        coeff = u.T @ x0
        x = u @ (np.exp(-float(gamma) * lam) * coeff)
        out.append(x.astype(np.float32))
    return np.stack(out, axis=0)


def estimate_gamma_knn_from_train(
    w: np.ndarray,
    fit_train_idx: np.ndarray,
    gamma_fit: np.ndarray,
    k_neighbors: int = 5,
    leave_one_out: bool = True,
) -> np.ndarray:
    n = w.shape[0]
    n_nodes = w.shape[-1]
    tri = np.triu_indices(n_nodes, k=1)

    z_all = w[:, tri[0], tri[1]].astype(np.float64)
    z_fit = z_all[fit_train_idx]
    mu = z_fit.mean(axis=0, keepdims=True)
    sd = z_fit.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    z_all = (z_all - mu) / sd
    z_fit = z_all[fit_train_idx]

    out = np.zeros((n,), dtype=np.float64)
    pos_map = {int(idx): p for p, idx in enumerate(fit_train_idx.tolist())}
    k_neighbors = max(1, int(k_neighbors))

    for i in range(n):
        d2 = np.sum((z_fit - z_all[i : i + 1]) ** 2, axis=1)
        if leave_one_out and i in pos_map and len(fit_train_idx) > 1:
            d2[pos_map[i]] = np.inf
        finite_mask = np.isfinite(d2)
        n_valid = int(np.sum(finite_mask))
        if n_valid == 0:
            out[i] = float(np.mean(gamma_fit))
            continue
        k_eff = min(k_neighbors, n_valid)
        nn = np.argpartition(d2, kth=k_eff - 1)[:k_eff]
        out[i] = float(np.mean(gamma_fit[nn]))
    return out


def mse_mae_r2(y_true: np.ndarray, y_pred: np.ndarray):
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    diff = yp - yt
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mse, mae, r2


def mean_cosine_similarity(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.ndim == 1:
        yt = yt[None, :]
        yp = yp[None, :]
    dot = np.sum(yt * yp, axis=1)
    n1 = np.sqrt(np.sum(yt * yt, axis=1))
    n2 = np.sqrt(np.sum(yp * yp, axis=1))
    cs = dot / (n1 * n2 + eps)
    return float(np.mean(cs))


def parse_roi_ids_1based_arg(s: str, n_rois: int) -> np.ndarray:
    if s is None:
        return np.array([], dtype=np.int32)
    out = []
    for x in s.split(","):
        t = x.strip()
        if not t:
            continue
        v = int(t)
        if 1 <= v <= n_rois:
            out.append(v - 1)
    if len(out) == 0:
        return np.array([], dtype=np.int32)
    return np.array(sorted(set(out)), dtype=np.int32)


def load_roi_labels(roi_labels_file: str, n_rois: int) -> List[str]:
    if roi_labels_file and os.path.exists(roi_labels_file):
        with open(roi_labels_file, "r", encoding="utf-8") as f:
            labels = [x.strip() for x in f if x.strip()]
        if len(labels) >= n_rois:
            return labels[:n_rois]
    return [f"ROI_{i+1:03d}" for i in range(n_rois)]


def infer_default_region_groups(roi_labels: List[str]) -> Dict[str, np.ndarray]:
    n = len(roi_labels)
    prefix = [x.split("_")[0] if "_" in x else x for x in roi_labels]

    mtl_prefix = {"Hipp", "PhG", "Amyg"}
    subcortical_prefix = {"BG", "Tha"}
    braak_limbic_prefix = {"CG", "ITG", "MTG", "STG", "FuG", "pSTS"}

    idx_mtl = np.array([i for i, p in enumerate(prefix) if p in mtl_prefix], dtype=np.int32)
    idx_limbic = np.array([i for i, p in enumerate(prefix) if p in braak_limbic_prefix], dtype=np.int32)
    idx_subcortical = np.array([i for i, p in enumerate(prefix) if p in subcortical_prefix], dtype=np.int32)
    mtl_set = set(idx_mtl.tolist())
    subcort_set = set(idx_subcortical.tolist())
    limbic_set = set(idx_limbic.tolist())
    idx_neocortical = np.array(
        [i for i in range(n) if i not in mtl_set and i not in subcort_set],
        dtype=np.int32,
    )
    idx_braak_v_vi = np.array(
        [i for i in idx_neocortical.tolist() if i not in limbic_set],
        dtype=np.int32,
    )

    return {
        "braak_i_ii": idx_mtl,
        "braak_iii_iv": idx_limbic,
        "braak_v_vi": idx_braak_v_vi,
        "mtl": idx_mtl,
        "neocortical": idx_neocortical,
    }


def build_region_groups(
    n_rois: int,
    roi_labels: List[str],
    braak_i_ii_rois: str,
    braak_iii_iv_rois: str,
    braak_v_vi_rois: str,
    mtl_rois: str,
    neo_rois: str,
):
    default_groups = infer_default_region_groups(roi_labels)
    groups = {}
    custom_flags = {}

    specs = {
        "braak_i_ii": braak_i_ii_rois,
        "braak_iii_iv": braak_iii_iv_rois,
        "braak_v_vi": braak_v_vi_rois,
        "mtl": mtl_rois,
        "neocortical": neo_rois,
    }
    for name, arg in specs.items():
        idx = parse_roi_ids_1based_arg(arg, n_rois=n_rois)
        if idx.size > 0:
            groups[name] = idx
            custom_flags[name] = True
        else:
            groups[name] = default_groups[name]
            custom_flags[name] = False
    return groups, custom_flags


def region_mse(y_true: np.ndarray, y_pred: np.ndarray, idx: np.ndarray) -> float:
    if idx is None or len(idx) == 0:
        return float("nan")
    yt = np.asarray(y_true, dtype=np.float64)[:, idx]
    yp = np.asarray(y_pred, dtype=np.float64)[:, idx]
    return float(np.mean((yp - yt) ** 2))


def compute_region_mse_metrics(
    y_train: np.ndarray,
    y_test: np.ndarray,
    pred_train: np.ndarray,
    pred_test: np.ndarray,
    region_groups: Dict[str, np.ndarray],
) -> Dict[str, float]:
    out = {}
    for name, idx in region_groups.items():
        out[f"train_mse_{name}"] = region_mse(y_train, pred_train, idx)
        out[f"test_mse_{name}"] = region_mse(y_test, pred_test, idx)
        out[f"n_rois_{name}"] = float(len(idx))
    return out


def compute_braak_region_mse(y_true: np.ndarray, y_pred: np.ndarray, region_groups: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {
        "braak_i_ii_mse": region_mse(y_true, y_pred, region_groups.get("braak_i_ii", np.array([], dtype=np.int32))),
        "braak_iii_iv_mse": region_mse(
            y_true, y_pred, region_groups.get("braak_iii_iv", np.array([], dtype=np.int32))
        ),
        "braak_v_vi_mse": region_mse(y_true, y_pred, region_groups.get("braak_v_vi", np.array([], dtype=np.int32))),
    }


def rankdata_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def safe_pearson_1d(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - np.mean(x)
    y = y - np.mean(y)
    den = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if den < eps:
        return float("nan")
    return float(np.sum(x * y) / den)


def mean_pearson_by_sample(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.ndim == 1:
        yt = yt[None, :]
        yp = yp[None, :]
    vals = [safe_pearson_1d(a, b) for a, b in zip(yt, yp)]
    vals = np.asarray(vals, dtype=np.float64)
    return float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else float("nan")


def mean_spearman_by_sample(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.ndim == 1:
        yt = yt[None, :]
        yp = yp[None, :]
    vals = []
    for a, b in zip(yt, yp):
        ra = rankdata_1d(a)
        rb = rankdata_1d(b)
        vals.append(safe_pearson_1d(ra, rb))
    vals = np.asarray(vals, dtype=np.float64)
    return float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else float("nan")


def compute_roi_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    diff = yp - yt
    mse = np.mean(diff ** 2, axis=0)
    mae = np.mean(np.abs(diff), axis=0)
    ss_res = np.sum((yt - yp) ** 2, axis=0)
    ss_tot = np.sum((yt - np.mean(yt, axis=0, keepdims=True)) ** 2, axis=0)
    r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)

    pearson = np.array([safe_pearson_1d(yt[:, j], yp[:, j]) for j in range(yt.shape[1])], dtype=np.float64)
    spearman = np.array(
        [safe_pearson_1d(rankdata_1d(yt[:, j]), rankdata_1d(yp[:, j])) for j in range(yt.shape[1])],
        dtype=np.float64,
    )
    return {"mse": mse, "mae": mae, "r2": r2, "pearson": pearson, "spearman": spearman}


def summarize_metric_array(x: np.ndarray, name: str, prefix: str) -> Dict[str, float]:
    arr = np.asarray(x, dtype=np.float64)
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return {
            f"{prefix}_{name}_mean": float("nan"),
            f"{prefix}_{name}_median": float("nan"),
            f"{prefix}_{name}_p10": float("nan"),
            f"{prefix}_{name}_p90": float("nan"),
            f"{prefix}_{name}_min": float("nan"),
            f"{prefix}_{name}_max": float("nan"),
            f"{prefix}_{name}_n_valid": 0.0,
        }
    return {
        f"{prefix}_{name}_mean": float(np.mean(valid)),
        f"{prefix}_{name}_median": float(np.median(valid)),
        f"{prefix}_{name}_p10": float(np.percentile(valid, 10)),
        f"{prefix}_{name}_p90": float(np.percentile(valid, 90)),
        f"{prefix}_{name}_min": float(np.min(valid)),
        f"{prefix}_{name}_max": float(np.max(valid)),
        f"{prefix}_{name}_n_valid": float(valid.size),
    }


class MLPRegressorTorch(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: List[int], dropout: float):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate_model(y_train, y_test, pred_train, pred_test, region_groups: Dict[str, np.ndarray] = None) -> Dict[str, float]:
    tr_mse, tr_mae, tr_r2 = mse_mae_r2(y_train, pred_train)
    te_mse, te_mae, te_r2 = mse_mae_r2(y_test, pred_test)
    tr_cs = mean_cosine_similarity(y_train, pred_train)
    te_cs = mean_cosine_similarity(y_test, pred_test)
    tr_pearson = mean_pearson_by_sample(y_train, pred_train)
    te_pearson = mean_pearson_by_sample(y_test, pred_test)
    tr_spearman = mean_spearman_by_sample(y_train, pred_train)
    te_spearman = mean_spearman_by_sample(y_test, pred_test)

    tr_roi = compute_roi_metrics(y_train, pred_train)
    te_roi = compute_roi_metrics(y_test, pred_test)

    out = {
        "train_mse": tr_mse,
        "train_mae": tr_mae,
        "train_r2": tr_r2,
        "train_cs": tr_cs,
        "train_pearson_by_sample": tr_pearson,
        "train_spearman_by_sample": tr_spearman,
        "test_mse": te_mse,
        "test_mae": te_mae,
        "test_r2": te_r2,
        "test_cs": te_cs,
        "test_pearson_by_sample": te_pearson,
        "test_spearman_by_sample": te_spearman,
    }
    for name in ["mse", "mae", "r2", "pearson", "spearman"]:
        out.update(summarize_metric_array(tr_roi[name], name=name, prefix="train_roi"))
        out.update(summarize_metric_array(te_roi[name], name=name, prefix="test_roi"))
        out.update(summarize_metric_array(tr_roi[name], name=f"{name}_by_roi", prefix="train_roi"))
        out.update(summarize_metric_array(te_roi[name], name=f"{name}_by_roi", prefix="test_roi"))

    if region_groups is not None and len(region_groups) > 0:
        out.update(compute_region_mse_metrics(y_train, y_test, pred_train, pred_test, region_groups))
    return out


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_name: str,
    mse_criterion: nn.Module,
    huber_criterion: nn.Module,
    cos_lambda: float,
) -> torch.Tensor:
    if loss_name == "mse":
        return mse_criterion(pred, target)
    if loss_name == "huber":
        return huber_criterion(pred, target)
    if loss_name == "mse_cos":
        mse = mse_criterion(pred, target)
        cos = nn.functional.cosine_similarity(pred, target, dim=1).mean()
        return mse + cos_lambda * (1.0 - cos)
    raise RuntimeError(f"Unsupported --loss {loss_name}")


def build_kfold_indices(n: int, n_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_folds < 2:
        raise RuntimeError("--cv-folds must be >= 2 in k-fold mode")
    if n_folds > n:
        raise RuntimeError(f"--cv-folds {n_folds} cannot exceed n_subjects {n}")
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.int32)
    rng.shuffle(idx)
    chunks = np.array_split(idx, n_folds)
    splits = []
    for i in range(n_folds):
        test_idx = np.sort(chunks[i].astype(np.int32))
        train_idx = np.sort(np.concatenate([chunks[j] for j in range(n_folds) if j != i]).astype(np.int32))
        if train_idx.size == 0 or test_idx.size == 0:
            raise RuntimeError("Invalid k-fold split: empty train/test fold")
        splits.append((train_idx, test_idx))
    return splits


def aggregate_metric_dicts(metric_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if len(metric_dicts) == 0:
        return {}
    keys = sorted(set().union(*[set(m.keys()) for m in metric_dicts]))
    out = {}
    for k in keys:
        vals = []
        for m in metric_dicts:
            v = m.get(k, float("nan"))
            try:
                vals.append(float(v))
            except Exception:
                vals.append(float("nan"))
        arr = np.asarray(vals, dtype=np.float64)
        valid = arr[~np.isnan(arr)]
        out[f"{k}_mean"] = float(np.mean(valid)) if valid.size > 0 else float("nan")
        out[f"{k}_std"] = float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0
    return out


def write_roi_metrics_csv(path: str, y_true: np.ndarray, pred_mean: np.ndarray, pred_mlp: np.ndarray):
    mean_test_roi = compute_roi_metrics(y_true, pred_mean)
    mlp_test_roi = compute_roi_metrics(y_true, pred_mlp)
    roi_rows = []
    for j in range(y_true.shape[1]):
        roi_rows.append(
            {
                "roi_1based": int(j + 1),
                "mean_mse": float(mean_test_roi["mse"][j]),
                "mean_mae": float(mean_test_roi["mae"][j]),
                "mean_r2": float(mean_test_roi["r2"][j]),
                "mean_pearson_by_roi": float(mean_test_roi["pearson"][j]),
                "mean_spearman_by_roi": float(mean_test_roi["spearman"][j]),
                "mlp_mse": float(mlp_test_roi["mse"][j]),
                "mlp_mae": float(mlp_test_roi["mae"][j]),
                "mlp_r2": float(mlp_test_roi["r2"][j]),
                "mlp_pearson_by_roi": float(mlp_test_roi["pearson"][j]),
                "mlp_spearman_by_roi": float(mlp_test_roi["spearman"][j]),
                "delta_r2_mlp_minus_mean": float(mlp_test_roi["r2"][j] - mean_test_roi["r2"][j]),
                "delta_mse_mlp_minus_mean": float(mlp_test_roi["mse"][j] - mean_test_roi["mse"][j]),
            }
        )
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(roi_rows[0].keys()))
        w.writeheader()
        w.writerows(roi_rows)


def train_one_fold(
    X_fit_train_n: np.ndarray,
    y_fit_train: np.ndarray,
    X_val_n: np.ndarray,
    y_val: np.ndarray,
    X_train_all_n: np.ndarray,
    X_test_n: np.ndarray,
    hidden: List[int],
    args,
    device: torch.device,
    fold_tag: str,
    huber_beta: float,
):
    model = MLPRegressorTorch(
        in_dim=X_fit_train_n.shape[1],
        out_dim=y_fit_train.shape[1],
        hidden=hidden,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse_criterion = nn.MSELoss()
    huber_criterion = nn.SmoothL1Loss(beta=huber_beta)

    ds_train = TensorDataset(torch.from_numpy(X_fit_train_n), torch.from_numpy(y_fit_train))
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=False)
    X_val_t = torch.from_numpy(X_val_n).to(device) if X_val_n.shape[0] > 0 else None
    y_val_t = torch.from_numpy(y_val).to(device) if X_val_n.shape[0] > 0 else None

    best = {"epoch": 0, "val_loss": float("inf"), "val_mse": float("inf"), "state_dict": None}
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_items = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = compute_loss(
                pred=pred,
                target=yb,
                loss_name=args.loss,
                mse_criterion=mse_criterion,
                huber_criterion=huber_criterion,
                cos_lambda=args.cos_lambda,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * xb.shape[0]
            train_items += int(xb.shape[0])

        train_loss_epoch = train_loss_sum / max(1, train_items)

        if X_val_n.shape[0] > 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = float(
                    compute_loss(
                        pred=val_pred,
                        target=y_val_t,
                        loss_name=args.loss,
                        mse_criterion=mse_criterion,
                        huber_criterion=huber_criterion,
                        cos_lambda=args.cos_lambda,
                    ).item()
                )
                val_mse = float(mse_criterion(val_pred, y_val_t).item())
        else:
            val_loss = train_loss_epoch
            val_mse = train_loss_epoch if args.loss == "mse" else float("nan")

        if val_loss < best["val_loss"]:
            best["epoch"] = epoch
            best["val_loss"] = val_loss
            best["val_mse"] = val_mse
            best["state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"{fold_tag} epoch {epoch:03d} train_loss={train_loss_epoch:.6f} "
                f"val_loss={val_loss:.6f} val_mse={val_mse:.6f}"
            )

        if patience_count >= args.patience:
            print(f"{fold_tag} Early stop at epoch {epoch} (patience={args.patience})")
            break

    if best["state_dict"] is None:
        raise RuntimeError("Training failed: best checkpoint is empty")

    best["huber_beta"] = float(huber_beta)
    model.load_state_dict(best["state_dict"])
    model.eval()
    with torch.no_grad():
        pred_train = model(torch.from_numpy(X_train_all_n).to(device)).cpu().numpy().astype(np.float32)
        pred_test = model(torch.from_numpy(X_test_n).to(device)).cpu().numpy().astype(np.float32)

    return pred_train, pred_test, best


def main():
    p = argparse.ArgumentParser(description="GPU MLP multi-output regression: full-network features -> 246 ROI SUVR")
    p.add_argument("--data", default="/mnt/disk3/ADNI_DTI_fMRI/Ana_Code/Model/GNN_Input/brainnectome_gnn_pet_tau_suvr.npz")
    p.add_argument("--pair-csv", default="/mnt/disk3/ADNI_DTI_fMRI/Ana_Code/PET_Axial_DTI_one_pair_per_subject.csv")
    p.add_argument("--out-dir", default="/mnt/disk3/ADNI_DTI_fMRI/Ana_Code/Model/GNN_Input/quick_regression_mlp_gpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.8, help="Used only when --cv-folds=1")
    p.add_argument("--cv-folds", type=int, default=5, help="Number of folds; set 1 to use holdout split")
    p.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio inside train split")
    p.add_argument("--edge-keys", default="A_count")
    p.add_argument("--node-keys", default="strength_count,mean_fa_nonzero,mean_rd_nonzero")
    p.add_argument("--only-dx", default="ALL", choices=["ALL", "CN", "MCI", "AD"])
    p.add_argument(
        "--roi-labels-file",
        default="/mnt/disk3/ADNI_DTI_fMRI/Ana_Code/SC_Brainnectome_Models/Brainnectome_labels_common.txt",
        help="ROI label file (one label per line) used for default Braak/MTL/Neocortical grouping",
    )
    p.add_argument("--braak-i-ii-rois", default="", help="Override Braak I/II ROI ids (1-based, comma-separated)")
    p.add_argument("--braak-iii-iv-rois", default="", help="Override Braak III/IV ROI ids (1-based, comma-separated)")
    p.add_argument("--braak-v-vi-rois", default="", help="Override Braak V/VI ROI ids (1-based, comma-separated)")
    p.add_argument("--mtl-rois", default="", help="Override medial temporal ROI ids (1-based, comma-separated)")
    p.add_argument("--neo-rois", default="", help="Override neocortical ROI ids (1-based, comma-separated)")
    p.add_argument("--ndm-enable", action="store_true")
    p.add_argument("--ndm-edge-key", default="A_count")
    p.add_argument("--ndm-seed-ids", default="115,116")
    p.add_argument("--ndm-beta-grid", default="0.05,0.1,0.2,0.5,1.0")
    p.add_argument("--ndm-time-grid", default="0.5,1.0,2.0,4.0,8.0")
    p.add_argument("--ndm-log1p-w", action="store_true")
    p.add_argument(
        "--ndm-personalized",
        action="store_true",
        help="Use personalized NDM gamma per subject (fit on train, infer for val/test by SC-kNN)",
    )
    p.add_argument(
        "--ndm-k-neighbors",
        type=int,
        default=5,
        help="k for SC-kNN gamma transfer when --ndm-personalized is enabled",
    )
    p.add_argument("--hidden", default="512,256")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--loss", default="mse", choices=["mse", "huber", "mse_cos"])
    p.add_argument("--huber-beta", type=float, default=1.0, help="delta/beta for SmoothL1Loss")
    p.add_argument(
        "--huber-beta-grid",
        default="",
        help="Auto-tune Huber beta on validation set; comma-separated (e.g., 0.3,0.5,1.0,2.0)",
    )
    p.add_argument("--cos-lambda", type=float, default=0.1, help="weight for cosine term in mse_cos")
    p.add_argument("--print-metrics-json", action="store_true", help="Print full metrics JSON to console")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    d = np.load(args.data, allow_pickle=True)
    subject_ids = d["subject_ids"]
    A = d["A"].astype(np.float32)
    X = d["X"].astype(np.float32)
    y = d["y_suvr"].astype(np.float32)

    n_subjects_before_dx_filter = int(subject_ids.shape[0])
    if args.only_dx != "ALL":
        sid_to_dx = {}
        with open(args.pair_csv, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                sid = (row.get("subject_id") or "").strip()
                dx = (row.get("pet_diagnosis_label") or "").strip().upper()
                if sid:
                    sid_to_dx[sid] = dx
        keep_idx = np.array(
            [i for i, sid in enumerate(subject_ids.tolist()) if sid_to_dx.get(str(sid), "") == args.only_dx],
            dtype=np.int32,
        )
        if keep_idx.size == 0:
            raise RuntimeError(f"No subjects left after --only-dx {args.only_dx}")
        subject_ids = subject_ids[keep_idx]
        A = A[keep_idx]
        X = X[keep_idx]
        y = y[keep_idx]

    edge_names = [str(x) for x in d["edge_feature_names"].tolist()]
    node_names = [str(x) for x in d["node_feature_names"].tolist()]
    use_edge_keys = parse_list_arg(args.edge_keys)
    use_node_keys = parse_list_arg(args.node_keys)

    n = int(A.shape[0])
    n_rois = int(y.shape[1])
    roi_labels = load_roi_labels(args.roi_labels_file, n_rois=n_rois)
    region_groups, region_custom_flags = build_region_groups(
        n_rois=n_rois,
        roi_labels=roi_labels,
        braak_i_ii_rois=args.braak_i_ii_rois,
        braak_iii_iv_rois=args.braak_iii_iv_rois,
        braak_v_vi_rois=args.braak_v_vi_rois,
        mtl_rois=args.mtl_rois,
        neo_rois=args.neo_rois,
    )

    if args.cv_folds <= 1:
        split_list = [split_indices(n=n, train_ratio=args.train_ratio, seed=args.seed)]
    else:
        split_list = build_kfold_indices(n=n, n_folds=args.cv_folds, seed=args.seed)
    cv_mode = len(split_list) > 1

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available")
        device = torch.device(args.device)

    hidden = [int(x.strip()) for x in args.hidden.split(",") if x.strip()]
    if len(hidden) == 0:
        raise RuntimeError("Empty --hidden. Example: --hidden 512,256")

    huber_beta_candidates = [float(args.huber_beta)]
    if args.loss == "huber":
        if args.huber_beta_grid.strip():
            huber_beta_candidates = parse_float_list_arg(args.huber_beta_grid)
            if len(huber_beta_candidates) == 0:
                raise RuntimeError("Empty --huber-beta-grid after parsing")
        for b in huber_beta_candidates:
            if b <= 0:
                raise RuntimeError("--huber-beta and --huber-beta-grid values must be > 0")

    ndm_evals_all = None
    ndm_evecs_all = None
    ndm_w_all = None
    ndm_x0 = None
    ndm_beta_grid = None
    ndm_time_grid = None
    ndm_seed_ids = None
    ndm_c = None
    feat_static = None

    if args.ndm_enable:
        if args.ndm_k_neighbors < 1:
            raise RuntimeError("--ndm-k-neighbors must be >= 1")
        if args.ndm_edge_key not in edge_names:
            raise RuntimeError(f"--ndm-edge-key {args.ndm_edge_key} not found in edge_feature_names")
        ndm_c = edge_names.index(args.ndm_edge_key)
        W = A[:, ndm_c].astype(np.float32)
        W = np.clip(W, a_min=0.0, a_max=None)
        if args.ndm_log1p_w:
            W = np.log1p(W)
        n_nodes = A.shape[-1]
        ndm_seed_ids = parse_int_list_arg(args.ndm_seed_ids)
        seed_ids0 = [s - 1 for s in ndm_seed_ids if 1 <= s <= n_nodes]
        if len(seed_ids0) == 0:
            raise RuntimeError("No valid seed ids in --ndm-seed-ids")
        ndm_x0 = np.zeros((n_nodes,), dtype=np.float32)
        ndm_x0[seed_ids0] = 1.0
        ndm_x0 /= np.sum(ndm_x0)

        ndm_beta_grid = parse_float_list_arg(args.ndm_beta_grid)
        ndm_time_grid = parse_float_list_arg(args.ndm_time_grid)
        if len(ndm_beta_grid) == 0 or len(ndm_time_grid) == 0:
            raise RuntimeError("Empty --ndm-beta-grid or --ndm-time-grid")
        ndm_w_all = W
        ndm_evals_all, ndm_evecs_all = ndm_precompute_eig(W)
    else:
        feat_static = build_full_network_features(
            A=A,
            X=X,
            edge_names=edge_names,
            node_names=node_names,
            use_edge_keys=use_edge_keys,
            use_node_keys=use_node_keys,
        )

    oof_pred_mlp = np.full_like(y, np.nan, dtype=np.float32)
    oof_pred_mean = np.full_like(y, np.nan, dtype=np.float32)
    oof_fold_ids = np.full((n,), -1, dtype=np.int32)
    fold_rows = []
    fold_test_metrics_mean = []
    fold_test_metrics_mlp = []
    holdout_cache = {}

    for fold_i, (train_idx, test_idx) in enumerate(split_list, start=1):
        fit_train_idx, val_idx = split_train_val(train_idx, val_ratio=args.val_ratio, seed=args.seed + fold_i * 101)
        fold_tag = f"[fold {fold_i}/{len(split_list)}]"
        print(
            f"{fold_tag} n_train={len(train_idx)} n_fit_train={len(fit_train_idx)} "
            f"n_val={len(val_idx)} n_test={len(test_idx)}"
        )

        ndm_info_fold = {"enabled": False}
        use_node_keys_fold = list(use_node_keys)
        if args.ndm_enable:
            evals_tr = [ndm_evals_all[i] for i in fit_train_idx]
            evecs_tr = [ndm_evecs_all[i] for i in fit_train_idx]
            if args.ndm_personalized:
                best_ndm_subj = fit_ndm_subject_params(
                    evals_train=evals_tr,
                    evecs_train=evecs_tr,
                    y_train=y[fit_train_idx],
                    x0=ndm_x0,
                    beta_grid=ndm_beta_grid,
                    time_grid=ndm_time_grid,
                )
                gamma_all = estimate_gamma_knn_from_train(
                    w=ndm_w_all,
                    fit_train_idx=fit_train_idx,
                    gamma_fit=best_ndm_subj["gamma"],
                    k_neighbors=args.ndm_k_neighbors,
                    leave_one_out=True,
                )
                ndm_all = ndm_predict_from_eig_subject_gammas(
                    ndm_evals_all,
                    ndm_evecs_all,
                    x0=ndm_x0,
                    gammas=gamma_all,
                )
            else:
                best_ndm = fit_ndm_global_params(
                    evals_train=evals_tr,
                    evecs_train=evecs_tr,
                    y_train=y[fit_train_idx],
                    x0=ndm_x0,
                    beta_grid=ndm_beta_grid,
                    time_grid=ndm_time_grid,
                )
                gamma_all = np.full((n,), float(best_ndm["gamma"]), dtype=np.float64)
                ndm_all = ndm_predict_from_eig(ndm_evals_all, ndm_evecs_all, x0=ndm_x0, gamma=best_ndm["gamma"])
            X_fold = np.concatenate([X, ndm_all[:, :, None].astype(np.float32)], axis=-1)
            node_names_fold = node_names + ["ndm_x_t"]
            if "ndm_x_t" not in use_node_keys_fold:
                use_node_keys_fold.append("ndm_x_t")

            feat = build_full_network_features(
                A=A,
                X=X_fold,
                edge_names=edge_names,
                node_names=node_names_fold,
                use_edge_keys=use_edge_keys,
                use_node_keys=use_node_keys_fold,
            )
            ndm_info_fold = {
                "enabled": True,
                "mode": "personalized_knn" if args.ndm_personalized else "global",
                "ndm_edge_key": args.ndm_edge_key,
                "seed_ids_1based": ndm_seed_ids,
                "beta_grid": ndm_beta_grid,
                "time_grid": ndm_time_grid,
                "ndm_fit_train_mse": float(np.mean((ndm_all[fit_train_idx] - y[fit_train_idx]) ** 2)),
                "ndm_test_mse": float(np.mean((ndm_all[test_idx] - y[test_idx]) ** 2)),
                "gamma_fit_train_mean": float(np.mean(gamma_all[fit_train_idx])),
                "gamma_fit_train_std": float(np.std(gamma_all[fit_train_idx], ddof=1))
                if len(fit_train_idx) > 1
                else 0.0,
                "gamma_test_mean": float(np.mean(gamma_all[test_idx])),
                "gamma_test_std": float(np.std(gamma_all[test_idx], ddof=1)) if len(test_idx) > 1 else 0.0,
                "k_neighbors": int(args.ndm_k_neighbors) if args.ndm_personalized else None,
                "log1p_w": bool(args.ndm_log1p_w),
                "feature_append": "ndm_x_t",
            }
            if args.ndm_personalized:
                ndm_info_fold.update(
                    {
                        "fit_subject_beta_mean": float(np.mean(best_ndm_subj["beta"])),
                        "fit_subject_t_mean": float(np.mean(best_ndm_subj["t"])),
                        "fit_subject_gamma_mean": float(np.mean(best_ndm_subj["gamma"])),
                        "fit_subject_mse_mean": float(np.mean(best_ndm_subj["mse"])),
                    }
                )
            else:
                ndm_info_fold.update(
                    {
                        "best_beta": float(best_ndm["beta"]),
                        "best_t": float(best_ndm["t"]),
                        "best_gamma": float(best_ndm["gamma"]),
                        "best_fit_train_mse": float(best_ndm["mse"]),
                    }
                )
        else:
            feat = feat_static

        X_fit_train = feat[fit_train_idx]
        X_val = feat[val_idx] if len(val_idx) > 0 else np.zeros((0, feat.shape[1]), dtype=np.float32)
        X_train_all = feat[train_idx]
        X_test = feat[test_idx]
        y_fit_train = y[fit_train_idx]
        y_val = y[val_idx] if len(val_idx) > 0 else np.zeros((0, y.shape[1]), dtype=np.float32)
        y_train_all = y[train_idx]
        y_test = y[test_idx]

        X_fit_train_n, X_val_n = standardize_train_test(X_fit_train, X_val)
        _, X_train_all_n = standardize_train_test(X_fit_train, X_train_all)
        _, X_test_n = standardize_train_test(X_fit_train, X_test)

        beta_trials = []
        if args.loss == "huber" and len(huber_beta_candidates) > 1:
            best_choice = None
            for beta_i, huber_beta in enumerate(huber_beta_candidates):
                set_seed(args.seed + fold_i * 1000 + beta_i)
                pred_train_i, pred_test_i, best_i = train_one_fold(
                    X_fit_train_n=X_fit_train_n,
                    y_fit_train=y_fit_train,
                    X_val_n=X_val_n,
                    y_val=y_val,
                    X_train_all_n=X_train_all_n,
                    X_test_n=X_test_n,
                    hidden=hidden,
                    args=args,
                    device=device,
                    fold_tag=f"{fold_tag} hb={huber_beta:g}",
                    huber_beta=float(huber_beta),
                )
                beta_trials.append(
                    {
                        "huber_beta": float(huber_beta),
                        "best_epoch": int(best_i["epoch"]),
                        "best_val_loss": float(best_i["val_loss"]),
                        "best_val_mse": float(best_i["val_mse"]),
                    }
                )
                if best_choice is None or best_i["val_loss"] < best_choice["best"]["val_loss"]:
                    best_choice = {"pred_train": pred_train_i, "pred_test": pred_test_i, "best": best_i}
            pred_train = best_choice["pred_train"]
            pred_test = best_choice["pred_test"]
            best = best_choice["best"]
            print(
                f"{fold_tag} selected_huber_beta={float(best['huber_beta']):.6g} "
                f"best_val_loss={float(best['val_loss']):.6f}"
            )
        else:
            set_seed(args.seed + fold_i)
            pred_train, pred_test, best = train_one_fold(
                X_fit_train_n=X_fit_train_n,
                y_fit_train=y_fit_train,
                X_val_n=X_val_n,
                y_val=y_val,
                X_train_all_n=X_train_all_n,
                X_test_n=X_test_n,
                hidden=hidden,
                args=args,
                device=device,
                fold_tag=fold_tag,
                huber_beta=float(huber_beta_candidates[0]),
            )

        y_mean = np.mean(y_train_all, axis=0).astype(np.float32)
        pred_train_mean = np.tile(y_mean[None, :], (y_train_all.shape[0], 1))
        pred_test_mean = np.tile(y_mean[None, :], (y_test.shape[0], 1))
        mean_metrics = evaluate_model(
            y_train_all, y_test, pred_train_mean, pred_test_mean, region_groups=region_groups
        )
        mlp_metrics = evaluate_model(
            y_train_all, y_test, pred_train, pred_test, region_groups=region_groups
        )

        oof_pred_mlp[test_idx] = pred_test
        oof_pred_mean[test_idx] = pred_test_mean
        oof_fold_ids[test_idx] = fold_i

        fold_rows.append(
            {
                "fold": int(fold_i),
                "n_train_total": int(len(train_idx)),
                "n_fit_train": int(len(fit_train_idx)),
                "n_val": int(len(val_idx)),
                "n_test": int(len(test_idx)),
                "n_features": int(feat.shape[1]),
                "best_epoch": int(best["epoch"]),
                "best_val_loss": float(best["val_loss"]),
                "best_val_mse": float(best["val_mse"]),
                "selected_huber_beta": float(best["huber_beta"]),
                "huber_beta_trials": beta_trials,
                "ndm": ndm_info_fold,
                "models": {
                    "mean_baseline": mean_metrics,
                    "mlp_torch": mlp_metrics,
                },
            }
        )
        fold_test_metrics_mean.append({k: float(v) for k, v in mean_metrics.items() if k.startswith("test_")})
        fold_test_metrics_mlp.append({k: float(v) for k, v in mlp_metrics.items() if k.startswith("test_")})

        if not cv_mode:
            holdout_cache = {
                "train_idx": train_idx,
                "fit_train_idx": fit_train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
                "y_test": y_test,
                "pred_test": pred_test,
                "pred_test_mean": pred_test_mean,
                "mean_metrics": mean_metrics,
                "mlp_metrics": mlp_metrics,
                "best": best,
                "ndm_info": ndm_info_fold,
                "n_features": int(feat.shape[1]),
            }

    if cv_mode:
        if np.any(np.isnan(oof_pred_mlp)) or np.any(np.isnan(oof_pred_mean)):
            raise RuntimeError("OOF prediction contains NaN or missing folds.")

        oof_mean_metrics = evaluate_model(y, y, oof_pred_mean, oof_pred_mean, region_groups=region_groups)
        oof_mlp_metrics = evaluate_model(y, y, oof_pred_mlp, oof_pred_mlp, region_groups=region_groups)
        fold_test_summary = {
            "mean_baseline": aggregate_metric_dicts(fold_test_metrics_mean),
            "mlp_torch": aggregate_metric_dicts(fold_test_metrics_mlp),
        }

        out_metrics = os.path.join(args.out_dir, "quick_tau_mlp_gpu_cv_metrics.json")
        out_pred = os.path.join(args.out_dir, "quick_tau_mlp_gpu_oof_predictions.npz")
        out_roi_csv = os.path.join(args.out_dir, "quick_tau_mlp_gpu_oof_roi_metrics.csv")

        metrics = {
            "seed": args.seed,
            "device": str(device),
            "cv_mode": True,
            "cv_folds": int(len(split_list)),
            "n_subjects": int(n),
            "n_subjects_before_dx_filter": n_subjects_before_dx_filter,
            "target": "tau_suvr_246roi",
            "only_dx": args.only_dx,
            "edge_keys": use_edge_keys,
            "node_keys": use_node_keys,
            "region_definitions": {
                "groups_1based": {k: [int(i + 1) for i in v.tolist()] for k, v in region_groups.items()},
                "custom_override": {k: bool(v) for k, v in region_custom_flags.items()},
                "roi_labels_file": args.roi_labels_file,
            },
            "feature_mode": "full_network",
            "n_features": int(fold_rows[0]["n_features"]),
            "n_outputs": int(y.shape[1]),
            "hidden": hidden,
            "dropout": float(args.dropout),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "patience": int(args.patience),
            "loss": args.loss,
            "huber_beta": float(args.huber_beta),
            "huber_beta_grid": [float(x) for x in huber_beta_candidates],
            "huber_beta_auto_tune": bool(args.loss == "huber" and len(huber_beta_candidates) > 1),
            "cos_lambda": float(args.cos_lambda),
            "ndm_personalized": bool(args.ndm_personalized),
            "ndm_k_neighbors": int(args.ndm_k_neighbors),
            "fold_test_summary": fold_test_summary,
            "models": {
                "mean_baseline": oof_mean_metrics,
                "mlp_torch": oof_mlp_metrics,
            },
            "folds": fold_rows,
            "outputs": {
                "metrics_json": out_metrics,
                "predictions_npz": out_pred,
                "test_roi_metrics_csv": out_roi_csv,
            },
        }

        with open(out_metrics, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        np.savez_compressed(
            out_pred,
            subject_ids=subject_ids,
            y_true=y,
            y_pred_mlp_oof=oof_pred_mlp,
            y_pred_mean_oof=oof_pred_mean,
            fold_indices=oof_fold_ids,
        )
        write_roi_metrics_csv(out_roi_csv, y_true=y, pred_mean=oof_pred_mean, pred_mlp=oof_pred_mlp)

        if args.print_metrics_json:
            print(json.dumps(metrics, indent=2))
        print(
            "summary:",
            json.dumps(
                {
                    "cv_folds": int(len(split_list)),
                    "huber_beta_auto_tune": bool(args.loss == "huber" and len(huber_beta_candidates) > 1),
                    "selected_huber_beta_mean": float(np.mean([r["selected_huber_beta"] for r in fold_rows])),
                    "selected_huber_beta_std": float(np.std([r["selected_huber_beta"] for r in fold_rows], ddof=1))
                    if len(fold_rows) > 1
                    else 0.0,
                    "mean_fold_test_mse": fold_test_summary["mean_baseline"].get("test_mse_mean", float("nan")),
                    "mean_fold_test_r2": fold_test_summary["mean_baseline"].get("test_r2_mean", float("nan")),
                    "mlp_fold_test_mse": fold_test_summary["mlp_torch"].get("test_mse_mean", float("nan")),
                    "mlp_fold_test_r2": fold_test_summary["mlp_torch"].get("test_r2_mean", float("nan")),
                    "mlp_oof_test_pearson_by_sample": oof_mlp_metrics.get(
                        "test_pearson_by_sample", float("nan")
                    ),
                    "mlp_oof_test_spearman_by_sample": oof_mlp_metrics.get(
                        "test_spearman_by_sample", float("nan")
                    ),
                    "mlp_oof_test_mse_mtl": oof_mlp_metrics.get("test_mse_mtl", float("nan")),
                    "mlp_oof_test_mse_neocortical": oof_mlp_metrics.get("test_mse_neocortical", float("nan")),
                    "mlp_oof_test_mse_braak_i_ii": oof_mlp_metrics.get("test_mse_braak_i_ii", float("nan")),
                    "mlp_oof_test_mse_braak_iii_iv": oof_mlp_metrics.get("test_mse_braak_iii_iv", float("nan")),
                    "mlp_oof_test_mse_braak_v_vi": oof_mlp_metrics.get("test_mse_braak_v_vi", float("nan")),
                },
                indent=2,
            ),
        )
        print("metrics:", out_metrics)
        print("predictions:", out_pred)
        print("test_roi_metrics:", out_roi_csv)
        return

    train_idx = holdout_cache["train_idx"]
    fit_train_idx = holdout_cache["fit_train_idx"]
    val_idx = holdout_cache["val_idx"]
    test_idx = holdout_cache["test_idx"]
    y_test = holdout_cache["y_test"]
    pred_test = holdout_cache["pred_test"]
    pred_test_mean = holdout_cache["pred_test_mean"]
    mean_metrics = holdout_cache["mean_metrics"]
    mlp_metrics = holdout_cache["mlp_metrics"]
    best = holdout_cache["best"]
    ndm_info = holdout_cache["ndm_info"]
    n_features = holdout_cache["n_features"]

    out_metrics = os.path.join(args.out_dir, "quick_tau_mlp_gpu_metrics.json")
    out_pred = os.path.join(args.out_dir, "quick_tau_mlp_gpu_test_predictions.npz")
    out_roi_csv = os.path.join(args.out_dir, "quick_tau_mlp_gpu_test_roi_metrics.csv")

    metrics = {
        "seed": args.seed,
        "device": str(device),
        "cv_mode": False,
        "cv_folds": 1,
        "n_subjects": int(n),
        "n_subjects_before_dx_filter": n_subjects_before_dx_filter,
        "n_train_total": int(len(train_idx)),
        "n_fit_train": int(len(fit_train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "target": "tau_suvr_246roi",
        "only_dx": args.only_dx,
        "edge_keys": use_edge_keys,
        "node_keys": use_node_keys,
        "region_definitions": {
            "groups_1based": {k: [int(i + 1) for i in v.tolist()] for k, v in region_groups.items()},
            "custom_override": {k: bool(v) for k, v in region_custom_flags.items()},
            "roi_labels_file": args.roi_labels_file,
        },
        "ndm": ndm_info,
        "feature_mode": "full_network",
        "n_features": int(n_features),
        "n_outputs": int(y.shape[1]),
        "hidden": hidden,
        "dropout": float(args.dropout),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "patience": int(args.patience),
        "loss": args.loss,
        "huber_beta": float(args.huber_beta),
        "huber_beta_grid": [float(x) for x in huber_beta_candidates],
        "huber_beta_auto_tune": bool(args.loss == "huber" and len(huber_beta_candidates) > 1),
        "cos_lambda": float(args.cos_lambda),
        "ndm_personalized": bool(args.ndm_personalized),
        "ndm_k_neighbors": int(args.ndm_k_neighbors),
        "best_epoch": int(best["epoch"]),
        "best_val_loss": float(best["val_loss"]),
        "best_val_mse": float(best["val_mse"]),
        "folds": fold_rows,
        "outputs": {
            "metrics_json": out_metrics,
            "predictions_npz": out_pred,
            "test_roi_metrics_csv": out_roi_csv,
        },
        "models": {
            "mean_baseline": mean_metrics,
            "mlp_torch": mlp_metrics,
        },
    }

    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    np.savez_compressed(
        out_pred,
        test_subject_ids=subject_ids[test_idx],
        y_true=y_test,
        y_pred_mlp=pred_test,
        y_pred_mean=pred_test_mean,
        test_indices=test_idx,
        train_indices=train_idx,
        fit_train_indices=fit_train_idx,
        val_indices=val_idx,
    )
    write_roi_metrics_csv(out_roi_csv, y_true=y_test, pred_mean=pred_test_mean, pred_mlp=pred_test)

    if args.print_metrics_json:
        print(json.dumps(metrics, indent=2))
    print(
        "summary:",
        json.dumps(
            {
                "best_epoch": metrics["best_epoch"],
                "best_val_loss": metrics["best_val_loss"],
                "huber_beta_auto_tune": bool(args.loss == "huber" and len(huber_beta_candidates) > 1),
                "selected_huber_beta": float(best["huber_beta"]),
                "mean_test_mse": metrics["models"]["mean_baseline"]["test_mse"],
                "mean_test_r2": metrics["models"]["mean_baseline"]["test_r2"],
                "mlp_test_mse": metrics["models"]["mlp_torch"]["test_mse"],
                "mlp_test_r2": metrics["models"]["mlp_torch"]["test_r2"],
                "mlp_test_pearson_by_sample": metrics["models"]["mlp_torch"].get(
                    "test_pearson_by_sample", float("nan")
                ),
                "mlp_test_spearman_by_sample": metrics["models"]["mlp_torch"].get(
                    "test_spearman_by_sample", float("nan")
                ),
                "mlp_test_mse_mtl": metrics["models"]["mlp_torch"].get("test_mse_mtl", float("nan")),
                "mlp_test_mse_neocortical": metrics["models"]["mlp_torch"].get("test_mse_neocortical", float("nan")),
                "mlp_test_mse_braak_i_ii": metrics["models"]["mlp_torch"].get("test_mse_braak_i_ii", float("nan")),
                "mlp_test_mse_braak_iii_iv": metrics["models"]["mlp_torch"].get(
                    "test_mse_braak_iii_iv", float("nan")
                ),
                "mlp_test_mse_braak_v_vi": metrics["models"]["mlp_torch"].get("test_mse_braak_v_vi", float("nan")),
            },
            indent=2,
        ),
    )
    print("metrics:", out_metrics)
    print("predictions:", out_pred)
    print("test_roi_metrics:", out_roi_csv)


if __name__ == "__main__":
    main()

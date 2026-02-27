#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# -------------------------
# IO helpers
# -------------------------
def read_lines(path: str) -> list[str]:
    with open(path, "r") as f:
        return [x.strip() for x in f if x.strip()]


def load_names(names_txt: str) -> np.ndarray:
    names = np.array(read_lines(names_txt), dtype=str)
    if names.size == 0:
        raise ValueError(f"Empty names_txt: {names_txt}")
    return names


def load_labels_1col(labels_tsv: str) -> np.ndarray:
    # your labels are usually 1-col
    df = pd.read_csv(labels_tsv, sep="\t", header=None)
    lab = df.iloc[:, 0].astype(int).to_numpy()
    return lab


def load_latent(latent_tsv: str) -> np.ndarray:
    Z = np.loadtxt(latent_tsv, delimiter="\t")
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    return Z.astype(np.float32)


def load_cna(cna_csv: str) -> pd.DataFrame:
    cna = pd.read_csv(cna_csv, index_col=0)
    cna.index = cna.index.astype(str)
    return cna


# -------------------------
# CNA burden + centering
# -------------------------
def center_cna(C: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return C
    if mode == "per_segment_median":
        med = np.median(C, axis=0, keepdims=True)
        return C - med
    if mode == "per_segment_mean":
        mu = np.mean(C, axis=0, keepdims=True)
        return C - mu
    raise ValueError(f"Unknown center_mode={mode}")


def cna_burden(C: np.ndarray, mode: str) -> np.ndarray:
    # returns (N,)
    if mode == "mean_abs":
        return np.mean(np.abs(C), axis=1)
    if mode == "mean_sq":
        return np.mean(C * C, axis=1)
    raise ValueError(f"Unknown cna_burden_mode={mode}")


def cluster_stat(x: np.ndarray, stat: str) -> float:
    if stat == "median":
        return float(np.median(x))
    if stat == "mean":
        return float(np.mean(x))
    raise ValueError(f"Unknown cluster_stat={stat}")


# -------------------------
# CNA-mixture cluster calling
# -------------------------
def fit_2comp_gmm_1d(x: np.ndarray, seed: int = 0) -> GaussianMixture:
    x = x.reshape(-1, 1).astype(np.float64)
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=seed)
    gmm.fit(x)
    return gmm


def tumor_component_from_gmm(gmm: GaussianMixture) -> int:
    # tumor component = higher mean
    means = gmm.means_.reshape(-1)
    return int(np.argmax(means))


def call_tumor_clusters_from_scores(
    cluster_scores: pd.DataFrame,
    tumor_posterior_thr: float,
    seed: int = 0,
) -> tuple[list[int], dict[int, float], dict[str, object]]:
    """
    cluster_scores must have: cluster, score
    returns tumor_clusters, posteriors, meta
    """
    x = cluster_scores["score"].to_numpy(dtype=np.float64)
    gmm = fit_2comp_gmm_1d(x, seed=seed)
    tumor_comp = tumor_component_from_gmm(gmm)

    post = gmm.predict_proba(x.reshape(-1, 1))[:, tumor_comp]
    post_map = {int(c): float(p) for c, p in zip(cluster_scores["cluster"], post)}

    tumor_clusters = [int(c) for c, p in post_map.items() if p >= tumor_posterior_thr]

    meta = {
        "gmm_means": gmm.means_.reshape(-1).tolist(),
        "gmm_covs": [float(c) for c in gmm.covariances_.reshape(-1)],
        "tumor_component": tumor_comp,
        "tumor_posterior_thr": float(tumor_posterior_thr),
    }
    return tumor_clusters, post_map, meta


# -------------------------
# Auto refinement (latent-only)
# -------------------------
def choose_k_by_silhouette(Z: np.ndarray, k_min: int, k_max: int, seed: int = 0) -> int:
    best_k = k_min
    best_s = -1.0
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Z)
        labels = km.labels_
        if len(np.unique(labels)) < 2:
            continue
        s = silhouette_score(Z, labels)
        if s > best_s:
            best_s = s
            best_k = k
    return int(best_k)


def refine_one_cluster_kmeans(
    Z_all: np.ndarray,
    labels: np.ndarray,
    target_cluster: int,
    k: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Split cells in target_cluster into k subclusters in latent space.
    Return new labels array with new IDs appended after max existing label.
    """
    mask = (labels == target_cluster)
    idx = np.where(mask)[0]
    Z = Z_all[idx]

    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Z)
    sub = km.labels_.astype(int)

    new_labels = labels.copy()
    start = int(new_labels.max()) + 1
    new_ids = start + sub

    new_labels[idx] = new_ids
    return new_labels


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--cna_csv", required=True)
    ap.add_argument("--names_txt", required=True)
    ap.add_argument("--labels_tsv", required=True)
    ap.add_argument("--latent_tsv", required=True)
    ap.add_argument("--out_prefix", required=True)

    ap.add_argument("--allow_missing_cna", action="store_true")

    # CNA burden config (this is your current best)
    ap.add_argument("--center_mode", default="per_segment_median",
                    choices=["per_segment_median", "per_segment_mean", "none"])
    ap.add_argument("--cna_burden_mode", default="mean_abs", choices=["mean_abs", "mean_sq"])
    ap.add_argument("--cluster_stat", default="median", choices=["median", "mean"])
    ap.add_argument("--log1p_cna", action="store_true")
    ap.add_argument("--tumor_posterior_thr", type=float, default=0.5)

    # Auto refine policy
    ap.add_argument("--auto_refine", action="store_true")
    ap.add_argument("--min_cluster_size_refine", type=int, default=500)
    ap.add_argument("--max_refine_clusters", type=int, default=3)
    ap.add_argument("--ambig_delta", type=float, default=0.12, help="Ambiguous band around posterior=0.5")
    ap.add_argument("--also_refine_big_normals", action="store_true",
                    help="If set, refine big clusters predicted normal even if not ambiguous. Recommended.")
    ap.add_argument("--topk_big_normals", type=int, default=1,
                    help="Refine up to top-K largest predicted-normal clusters (size>=min).")

    # K selection for refinement
    ap.add_argument("--k_auto", action="store_true")
    ap.add_argument("--k_min", type=int, default=2)
    ap.add_argument("--k_max", type=int, default=10)
    ap.add_argument("--k_fixed", type=int, default=2)

    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    seed = int(args.seed)
    np.random.seed(seed)

    # Load
    names = load_names(args.names_txt)
    labels = load_labels_1col(args.labels_tsv)
    Z = load_latent(args.latent_tsv)

    if not (len(names) == len(labels) == Z.shape[0]):
        m = min(len(names), len(labels), Z.shape[0])
        names = names[:m]
        labels = labels[:m]
        Z = Z[:m]
        print(f"[WARN] length mismatch; truncated to m={m}")

    cna_df = load_cna(args.cna_csv)

    # Align CNA to names order (intersection if allowed)
    have = set(cna_df.index)
    want = list(names)
    mask_have = np.array([n in have for n in want], dtype=bool)
    if not mask_have.all():
        missing = int((~mask_have).sum())
        if not args.allow_missing_cna:
            ex = [want[i] for i in np.where(~mask_have)[0][:5]]
            raise ValueError(f"CNA CSV missing {missing} cell_names (example: {ex}). "
                             f"Re-run with --allow_missing_cna.")
        # intersect: drop missing from all arrays
        names = names[mask_have]
        labels = labels[mask_have]
        Z = Z[mask_have]
        want = list(names)

    CNA = cna_df.loc[want].to_numpy(dtype=np.float32, copy=True)

    # center + burden
    CNA = center_cna(CNA, args.center_mode)
    b = cna_burden(CNA, args.cna_burden_mode).astype(np.float64)
    if args.log1p_cna:
        b = np.log1p(b)

    # per-cell table
    per_cell = pd.DataFrame({
        "cell_name": names,
        "cluster": labels.astype(int),
        "cna_burden": b,
    })

    # cluster-level scores (median/mean burden)
    cs = (
        per_cell.groupby("cluster")["cna_burden"]
        .apply(lambda x: cluster_stat(x.to_numpy(), args.cluster_stat))
        .reset_index()
        .rename(columns={"cna_burden": "score"})
        .sort_values("cluster")
        .reset_index(drop=True)
    )
    sizes = per_cell["cluster"].value_counts().to_dict()
    cs["size"] = cs["cluster"].map(lambda c: int(sizes.get(int(c), 0)))

    # initial CNA-mixture call
    tumor_clusters, post_map, gmm_meta = call_tumor_clusters_from_scores(
        cs[["cluster", "score"]].copy(),
        tumor_posterior_thr=args.tumor_posterior_thr,
        seed=seed,
    )
    cs["tumor_posterior"] = cs["cluster"].map(lambda c: float(post_map[int(c)]))

    # auto refinement selection
    refined = False
    refine_targets: list[int] = []

    if args.auto_refine:
        # ambiguous clusters around 0.5 posterior
        amb_lo = 0.5 - float(args.ambig_delta)
        amb_hi = 0.5 + float(args.ambig_delta)

        ambiguous = cs[
            (cs["size"] >= args.min_cluster_size_refine)
            & (cs["tumor_posterior"] >= amb_lo)
            & (cs["tumor_posterior"] <= amb_hi)
        ]["cluster"].astype(int).tolist()

        refine_targets.extend(ambiguous)

        # optionally refine big predicted-normal clusters (this is what fixes Gao)
        if args.also_refine_big_normals:
            normals = cs[
                (cs["size"] >= args.min_cluster_size_refine)
                & (cs["tumor_posterior"] < args.tumor_posterior_thr)
            ].copy()
            normals = normals.sort_values("size", ascending=False)
            refine_targets.extend(normals["cluster"].astype(int).tolist()[: int(args.topk_big_normals)])

        # de-dup while keeping order
        seen = set()
        refine_targets = [c for c in refine_targets if not (c in seen or seen.add(c))]

        refine_targets = refine_targets[: int(args.max_refine_clusters)]

        # execute refinement
        if len(refine_targets) > 0:
            new_labels = labels.copy()
            for c in refine_targets:
                idx = np.where(new_labels == c)[0]
                if idx.size < args.min_cluster_size_refine:
                    continue

                Zc = Z[idx]
                if args.k_auto:
                    k = choose_k_by_silhouette(Zc, args.k_min, args.k_max, seed=seed)
                else:
                    k = int(args.k_fixed)

                if k <= 1:
                    continue

                new_labels = refine_one_cluster_kmeans(Z, new_labels, c, k=k, seed=seed)
                refined = True

            labels = new_labels

    # Recompute scores and final CNA-mixture call
    per_cell["cluster"] = labels.astype(int)

    cs2 = (
        per_cell.groupby("cluster")["cna_burden"]
        .apply(lambda x: cluster_stat(x.to_numpy(), args.cluster_stat))
        .reset_index()
        .rename(columns={"cna_burden": "score"})
        .sort_values("cluster")
        .reset_index(drop=True)
    )
    sizes2 = per_cell["cluster"].value_counts().to_dict()
    cs2["size"] = cs2["cluster"].map(lambda c: int(sizes2.get(int(c), 0)))

    tumor_clusters2, post_map2, gmm_meta2 = call_tumor_clusters_from_scores(
        cs2[["cluster", "score"]].copy(),
        tumor_posterior_thr=args.tumor_posterior_thr,
        seed=seed,
    )
    cs2["tumor_posterior"] = cs2["cluster"].map(lambda c: float(post_map2[int(c)]))

    # write outputs
    out_labels = f"{args.out_prefix}.labels.tsv"
    out_tumor = f"{args.out_prefix}.tumor_clusters.txt"
    out_scores = f"{args.out_prefix}.cluster_scores.tsv"
    out_percell = f"{args.out_prefix}.per_cell.tsv"
    out_meta = f"{args.out_prefix}.meta.json"

    pd.Series(labels.astype(int)).to_csv(out_labels, sep="\t", header=False, index=False)

    with open(out_tumor, "w") as f:
        for c in sorted(set(tumor_clusters2)):
            f.write(str(int(c)) + "\n")

    cs2.to_csv(out_scores, sep="\t", index=False)
    per_cell.to_csv(out_percell, sep="\t", index=False)

    meta = {
        "seed": seed,
        "aligned_cells_used": int(len(names)),
        "cna_segments": int(CNA.shape[1]),
        "refined": bool(refined),
        "refine_targets": refine_targets,
        "initial_gmm": gmm_meta,
        "final_gmm": gmm_meta2,
        "tumor_clusters": sorted(set(int(x) for x in tumor_clusters2)),
        "args": vars(args),
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Aligned cells used: N={len(names)}  CNA_segments={CNA.shape[1]}  latent_dim={Z.shape[1]}")
    print(f"[INFO] labels unique={len(np.unique(load_labels_1col(args.labels_tsv)))} -> final_unique={len(np.unique(labels))}")
    if args.auto_refine:
        print(f"[INFO] refined={refined}  refine_targets={refine_targets}")
    print("[OK] wrote:")
    print(" ", out_labels)
    print(" ", out_tumor)
    print(" ", out_scores)
    print(" ", out_percell)
    print(" ", out_meta)
    print("[INFO] tumor_clusters =", sorted(set(int(x) for x in tumor_clusters2)))


if __name__ == "__main__":
    main()
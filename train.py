from __future__ import annotations
import argparse
import json
import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

import utils
from models import GAT_AE


def scipy_to_tf_sparse(A: sp.csr_matrix) -> tf.sparse.SparseTensor:
    A = A.tocoo()
    idx = np.vstack([A.row, A.col]).T.astype(np.int64)
    vals = A.data.astype(np.float32)
    shape = np.array(A.shape, dtype=np.int64)
    return tf.sparse.SparseTensor(indices=idx, values=vals, dense_shape=shape)


def load_graph_npz(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    keys = set(d.files)

    # --- features ---
    if "rna_feat" in keys:
        X = d["rna_feat"]
    elif "x" in keys:
        X = d["x"]
    else:
        raise KeyError(f"NPZ missing RNA features. Expected 'rna_feat' or 'x'. Found keys={sorted(keys)}")

    # --- edges ---
    if "edge_index" not in keys:
        raise KeyError(f"NPZ missing 'edge_index'. Found keys={sorted(keys)}")
    edge_index = d["edge_index"]

    if "edge_weight" in keys:
        edge_weight = d["edge_weight"]
    elif "edge_w" in keys:
        edge_weight = d["edge_w"]
    else:
        raise KeyError(f"NPZ missing edge weights. Expected 'edge_weight' (or 'edge_w'). Found keys={sorted(keys)}")

    # --- names / params ---
    if "cell_names" in keys:
        cell_names = d["cell_names"].astype(str)
    elif "cells" in keys:
        cell_names = d["cells"].astype(str)
    else:
        cell_names = None

    params = None
    if "params" in keys:
        p = d["params"]
        try:
            if isinstance(p, np.ndarray) and p.shape == ():
                p = p.item()
            if isinstance(p, (bytes, str)):
                params = json.loads(p.decode() if isinstance(p, bytes) else p)
            elif isinstance(p, dict):
                params = p
            else:
                params = json.loads(str(p))
        except Exception:
            params = {"raw_params": str(p)}

    return X, edge_index, edge_weight, cell_names, params


def edges_to_csr(edge_index: np.ndarray, edge_weight: np.ndarray, n_nodes: int) -> sp.csr_matrix:
    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must be shape (2,E), got {edge_index.shape}")
    row = edge_index[0].astype(np.int64)
    col = edge_index[1].astype(np.int64)
    val = edge_weight.astype(np.float32)
    A = sp.csr_matrix((val, (row, col)), shape=(n_nodes, n_nodes), dtype=np.float32)
    return A


def build_one_pos_neighbor(edge_index: np.ndarray, n_nodes: int, seed: int = 42) -> np.ndarray:
    """
    For each node i, pick ONE positive neighbor pos[i] from outgoing edges.
    If a node has no neighbor (shouldn't happen with kNN), fall back to itself.
    """
    rng = np.random.RandomState(seed)
    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)

    neigh = [[] for _ in range(n_nodes)]
    for s, t in zip(src, dst):
        if s != t:
            neigh[s].append(t)

    pos = np.arange(n_nodes, dtype=np.int64)
    for i in range(n_nodes):
        if len(neigh[i]) > 0:
            pos[i] = neigh[i][rng.randint(len(neigh[i]))]
        else:
            pos[i] = i
    return pos


# ------------------------------
# Community clustering helpers
# ------------------------------

def knn_graph_from_latent(
    Z: np.ndarray,
    k: int = 30,
    metric: str = "euclidean",
    symmetrize: bool = True,
    weight_mode: str = "connectivity",  # "connectivity" or "distance"
) -> sp.csr_matrix:
    """
    Build a sparse kNN graph from latent embedding Z (N,D).
    Returns CSR adjacency (N,N) with weights.
    """
    N = Z.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, N), metric=metric)
    nn.fit(Z)
    dist, idx = nn.kneighbors(Z, return_distance=True)

    # drop self
    idx = idx[:, 1:]
    dist = dist[:, 1:]

    rows = np.repeat(np.arange(N, dtype=np.int64), idx.shape[1])
    cols = idx.reshape(-1).astype(np.int64)

    if weight_mode == "connectivity":
        vals = np.ones_like(cols, dtype=np.float32)
    elif weight_mode == "distance":
        # convert distance to similarity with a safe transform
        # (smaller dist => larger weight)
        d = dist.reshape(-1).astype(np.float32)
        vals = 1.0 / (1.0 + d)
    else:
        raise ValueError(f"Unknown weight_mode={weight_mode}")

    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N), dtype=np.float32)

    if symmetrize:
        # keep max weight for undirected
        A = A.maximum(A.T)

    # remove explicit self loops; community methods don't need them
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def cluster_leiden_from_adj(A: sp.csr_matrix, resolution: float, seed: int = 42) -> np.ndarray:
    """
    Preferred: leidenalg + igraph.
    Falls back by raising ImportError if unavailable.
    """
    import igraph as ig
    import leidenalg

    A = A.tocoo()
    edges = list(zip(A.row.tolist(), A.col.tolist()))
    weights = A.data.astype(float).tolist()

    g = ig.Graph(n=A.shape[0], edges=edges, directed=False)
    g.es["weight"] = weights

    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(resolution),
        seed=int(seed),
    )
    labels = np.array(part.membership, dtype=int)
    return labels


def cluster_scanpy_leiden(Z: np.ndarray, knn_k: int, resolution: float, seed: int = 42) -> np.ndarray:
    """
    Alternative: scanpy if installed.
    """
    import scanpy as sc

    adata = sc.AnnData(X=Z.astype(np.float32))
    sc.pp.neighbors(adata, n_neighbors=int(knn_k), use_rep="X")
    sc.tl.leiden(adata, resolution=float(resolution), random_state=int(seed))
    labels = adata.obs["leiden"].astype(int).to_numpy()
    return labels


def cluster_louvain_networkx(A: sp.csr_matrix, resolution: float, seed: int = 42) -> np.ndarray:
    """
    Fallback: python-louvain (community) + networkx.
    Note: resolution support depends on package version; we try to pass it.
    """
    import networkx as nx
    import community as community_louvain  # python-louvain

    A = A.tocoo()
    G = nx.Graph()
    G.add_nodes_from(range(A.shape[0]))
    for i, j, w in zip(A.row, A.col, A.data):
        if i == j:
            continue
        G.add_edge(int(i), int(j), weight=float(w))

    # community_louvain.best_partition may accept resolution=...
    try:
        part = community_louvain.best_partition(G, weight="weight", random_state=int(seed), resolution=float(resolution))
    except TypeError:
        part = community_louvain.best_partition(G, weight="weight", random_state=int(seed))

    labels = np.array([part[i] for i in range(A.shape[0])], dtype=int)
    return labels


def relabel_to_compact(labels: np.ndarray) -> np.ndarray:
    """
    Remap labels to 0..K-1 for cleanliness.
    """
    uniq = np.unique(labels)
    mp = {u: i for i, u in enumerate(uniq)}
    return np.array([mp[x] for x in labels], dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)

    ap.add_argument("--n_clusters", type=int, default=10)  # used by kmeans/gmm/spectral, ignored by leiden
    ap.add_argument("--hidden1", type=int, default=64)
    ap.add_argument("--hidden2", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--clusterer",
        default="leiden",
        choices=["leiden", "louvain", "hdbscan", "spectral", "agglom", "kmeans", "gmm"],
        help="Clustering method. Default: leiden (community-based, no fixed K).",
    )
    ap.add_argument("--out_prefix", default="out_rna")

    # Contrastive anti-collapse
    ap.add_argument("--lambda_contrast", type=float, default=1.0,
                    help="Weight for graph contrastive (InfoNCE) loss. Set 0 to disable.")
    ap.add_argument("--n_neg", type=int, default=256,
                    help="Number of negatives per node for InfoNCE.")
    ap.add_argument("--temperature", type=float, default=0.2,
                    help="InfoNCE temperature.")
    ap.add_argument("--zscore_features", action="store_true",
                    help="Z-score features per gene/feature (after loading).")
    ap.add_argument("--lambda_l2", type=float, default=0.0,
                    help="Optional L2 weight decay on embeddings (small value like 1e-4).")

    # Community clustering knobs (latent graph)
    ap.add_argument("--latent_knn", type=int, default=30, help="k for kNN graph built on latent (Leiden/Louvain/etc.)")
    ap.add_argument("--latent_metric", default="euclidean", choices=["euclidean", "cosine"], help="metric for latent kNN")
    ap.add_argument("--latent_weight_mode", default="connectivity", choices=["connectivity", "distance"],
                    help="Edge weights for latent kNN graph.")
    ap.add_argument("--resolution", type=float, default=1.0,
                    help="Community resolution (higher => more clusters). Used by Leiden/Louvain. "
                         "Overclustering recommended (e.g., 1.5-4.0).")

    args = ap.parse_args()

    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)

    X_np, edge_index, edge_weight, cell_names, params = load_graph_npz(args.npz)
    X_np = np.asarray(X_np, dtype=np.float32)
    n_cells, n_feats = X_np.shape

    if args.zscore_features:
        mu = X_np.mean(axis=0, keepdims=True)
        sd = X_np.std(axis=0, keepdims=True) + 1e-6
        X_np = (X_np - mu) / sd

    print(f"[INFO] Loaded NPZ: {args.npz}")
    print(f"[INFO] X: {X_np.shape} min/max={X_np.min():.4f}/{X_np.max():.4f}")
    print(f"[INFO] edges: {edge_index.shape[1]} weight min/max={edge_weight.min():.4f}/{edge_weight.max():.4f}")
    if params is not None:
        print(f"[INFO] npz params: {params}")

    # adjacency for model
    A = edges_to_csr(edge_index, edge_weight, n_nodes=n_cells)
    A_norm = utils.normalize_graph(A.copy(), normalized=True, add_self_loops=True)

    features = tf.convert_to_tensor(X_np, dtype=tf.float32)
    graph = scipy_to_tf_sparse(A)
    graph_norm = scipy_to_tf_sparse(A_norm)

    # fixed positive neighbor per node
    pos_idx_np = build_one_pos_neighbor(edge_index, n_nodes=n_cells, seed=seed)
    pos_idx = tf.constant(pos_idx_np, dtype=tf.int32)

    model = GAT_AE(in_dim=n_feats, hidden1=args.hidden1, hidden2=args.hidden2, dropout_rate=args.dropout)
    opt = tf.keras.optimizers.Adam(args.lr)

    @tf.function
    def info_nce_loss(Z: tf.Tensor) -> tf.Tensor:
        Z = tf.math.l2_normalize(Z, axis=1)
        Z_pos = tf.gather(Z, pos_idx)  # (N,D)
        logit_pos = tf.reduce_sum(Z * Z_pos, axis=1, keepdims=True) / args.temperature  # (N,1)

        neg_idx = tf.random.uniform(shape=(n_cells, args.n_neg), minval=0, maxval=n_cells, dtype=tf.int32)
        Z_neg = tf.gather(Z, neg_idx)  # (N,K,D)
        logit_neg = tf.einsum("nd,nkd->nk", Z, Z_neg) / args.temperature  # (N,K)

        logits = tf.concat([logit_pos, logit_neg], axis=1)  # (N, 1+K)
        labels = tf.zeros((n_cells,), dtype=tf.int32)       # correct class is 0
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        return loss

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            latent, _, recon = model([features, graph_norm, graph], training=True)
            mse = tf.reduce_mean(tf.square(features - recon))

            c_loss = tf.constant(0.0, tf.float32)
            if args.lambda_contrast > 0.0:
                c_loss = info_nce_loss(latent)

            l2 = tf.constant(0.0, tf.float32)
            if args.lambda_l2 > 0.0:
                l2 = tf.reduce_mean(tf.reduce_sum(tf.square(latent), axis=1))

            loss = mse + args.lambda_contrast * c_loss + args.lambda_l2 * l2

        grads = tape.gradient(loss, model.trainable_variables)
        grads = [tf.clip_by_norm(g, 5.0) if g is not None else None for g in grads]
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss, mse, c_loss, latent

    def cluster_latent(Z: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Returns (labels, info_dict).
        """
        info = {"clusterer": args.clusterer}

        if args.clusterer in ("leiden", "louvain"):
            A_lat = knn_graph_from_latent(
                Z, k=args.latent_knn, metric=args.latent_metric,
                symmetrize=True, weight_mode=args.latent_weight_mode
            )
            info.update({
                "latent_knn": int(args.latent_knn),
                "latent_metric": args.latent_metric,
                "latent_weight_mode": args.latent_weight_mode,
                "resolution": float(args.resolution),
                "latent_edges": int(A_lat.nnz // 2) if A_lat.nnz else int(A_lat.nnz),
            })

            if args.clusterer == "leiden":
                # try leidenalg/igraph first
                try:
                    labels = cluster_leiden_from_adj(A_lat, resolution=args.resolution, seed=seed)
                    labels = relabel_to_compact(labels)
                    info["backend"] = "leidenalg+igraph"
                    print("inside first try ")
                    return labels, info
                except Exception as e1:
                    # try scanpy leiden
                    print(e1)
                    try:
                        labels = cluster_scanpy_leiden(Z, knn_k=args.latent_knn, resolution=args.resolution, seed=seed)
                        labels = relabel_to_compact(labels)
                        info["backend"] = "scanpy"
                        info["warn_fallback"] = f"leidenalg failed: {type(e1).__name__}: {e1}"
                        return labels, info
                    except Exception as e2:
                        # fall back to louvain if possible
                        print(e2)
                        try:
                            labels = cluster_louvain_networkx(A_lat, resolution=args.resolution, seed=seed)
                            labels = relabel_to_compact(labels)
                            info["backend"] = "python-louvain+networkx"
                            info["warn_fallback"] = (
                                f"leiden backends failed: ({type(e1).__name__}) {e1}; "
                                f"scanpy failed: ({type(e2).__name__}) {e2}"
                            )
                            return labels, info
                        except Exception as e3:
                            print(e3)
                            info["backend"] = "fallback-spectral"
                            info["warn_fallback"] = (
                                f"community clustering failed: ({type(e1).__name__}) {e1}; "
                                f"scanpy: ({type(e2).__name__}) {e2}; "
                                f"louvain: ({type(e3).__name__}) {e3}"
                            )
                            # fall through to spectral below

        if args.clusterer == "hdbscan":
            try:
                import hdbscan
                cl = hdbscan.HDBSCAN(min_cluster_size=max(10, int(0.005 * len(Z))), metric="euclidean")
                labels = cl.fit_predict(Z)
                # hdbscan uses -1 for noise; keep it but compact for saving if you want:
                # here we keep -1 as is and remap others to 0..K-1
                noise = labels == -1
                labels2 = labels.copy()
                if np.any(~noise):
                    uniq = np.unique(labels2[~noise])
                    mp = {u: i for i, u in enumerate(uniq)}
                    labels2[~noise] = np.array([mp[x] for x in labels2[~noise]], dtype=int)
                info["backend"] = "hdbscan"
                return labels2.astype(int), info
            except Exception as e:
                info["backend"] = "fallback-spectral"
                info["warn_fallback"] = f"hdbscan failed: {type(e).__name__}: {e}"
                # fall through

        if args.clusterer == "spectral":
            cl = SpectralClustering(
                n_clusters=args.n_clusters,
                affinity="nearest_neighbors",
                n_neighbors=min(args.latent_knn, len(Z) - 1),
                random_state=seed,
                assign_labels="kmeans",
            )
            labels = cl.fit_predict(Z).astype(int)
            info["backend"] = "sklearn-spectral"
            return labels, info

        if args.clusterer == "agglom":
            # connectivity constrained agglomerative clustering
            A_lat = knn_graph_from_latent(
                Z, k=args.latent_knn, metric=args.latent_metric,
                symmetrize=True, weight_mode="connectivity"
            )
            cl = AgglomerativeClustering(n_clusters=args.n_clusters, connectivity=A_lat)
            labels = cl.fit_predict(Z).astype(int)
            info["backend"] = "sklearn-agglom"
            return labels, info

        if args.clusterer == "kmeans":
            cl = KMeans(n_clusters=args.n_clusters, random_state=seed, n_init=10).fit(Z)
            info["backend"] = "sklearn-kmeans"
            return cl.labels_.astype(int), info

        # gmm default fallback
        cl = GaussianMixture(n_components=args.n_clusters, covariance_type="full", random_state=seed).fit(Z)
        info["backend"] = "sklearn-gmm"
        return cl.predict(Z).astype(int), info

    best_sil = -1.0
    best_labels = None
    best_latent = None
    best_cluster_info = None

    for ep in range(args.epochs):
        loss, mse, c_loss, latent = train_step()

        if ep % 10 == 0 or ep == args.epochs - 1:
            Z = latent.numpy()
            zvar = float(np.median(Z.var(axis=0)))

            labels, cl_info = cluster_latent(Z)
            n_unique = len(set(labels.tolist()))
            sil = silhouette_score(Z, labels) if n_unique > 1 else -1.0

            if sil > best_sil:
                best_sil = sil
                best_labels = labels.copy()
                best_latent = Z.copy()
                best_cluster_info = cl_info

            print(
                f"epoch={ep:4d} loss={float(loss):.6f} mse={float(mse):.6f} "
                f"contrast={float(c_loss):.6f} zvar_med={zvar:.3e} "
                f"nclust={n_unique} sil={sil:.4f} best={best_sil:.4f} "
                f"clusterer={cl_info.get('backend','?')}"
            )
            if "warn_fallback" in cl_info:
                print(f"[WARN] {cl_info['warn_fallback']}")

    np.savetxt(f"{args.out_prefix}.latent.tsv", best_latent, delimiter="\t")
    np.savetxt(f"{args.out_prefix}.labels.tsv", best_labels.astype(int), fmt="%d")

    if cell_names is not None:
        with open(f"{args.out_prefix}.cells.txt", "w") as f:
            for c in cell_names:
                f.write(str(c) + "\n")

    meta = {
        "npz": args.npz,
        "n_cells": int(n_cells),
        "n_features": int(n_feats),
        "n_edges": int(edge_index.shape[1]),
        "best_silhouette": float(best_sil),
        "args": vars(args),
        "npz_params": params,
        "best_cluster_info": best_cluster_info,
        "n_clusters_best": int(len(np.unique(best_labels))) if best_labels is not None else None,
    }
    with open(f"{args.out_prefix}.meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("[OK] Saved:")
    print(f"  {args.out_prefix}.latent.tsv")
    print(f"  {args.out_prefix}.labels.tsv")
    if cell_names is not None:
        print(f"  {args.out_prefix}.cells.txt")
    print(f"  {args.out_prefix}.meta.json")


if __name__ == "__main__":
    main()


## Model Architecture

The model (`models.py`) is a **Graph Attention Autoencoder (GAT-AE)** built in TensorFlow/Keras.

**Encoder**
- `GATConv(in_dim → hidden1, attention=True, activation=ELU)` — learns per-edge attention weights from source and destination node projections
- `GATConv(hidden1 → hidden2, attention=False, activation=None)` — produces the latent embedding

**Decoder** (tied weights — decoder kernels are the transpose of encoder kernels)
- `GATConv(hidden2 → hidden1, attention=True, tied_attention=encoder_layer1.attentions, activation=ELU)`
- `GATConv(hidden1 → in_dim, attention=False, activation=None)` — reconstruction

The `GATConv` layer (`gat_conv.py`) computes attention scores as a sum of per-node source and destination projections, masks them with the sparse adjacency matrix, applies LeakyReLU (negative slope=0.2), then softmax-normalises per node before aggregating neighbour features.

**Training objective** combines:
- Reconstruction loss (MSE between input RNA features and decoded output)
- Supervised contrastive loss on the latent space (InfoNCE-style with random negative sampling)

---

## Repository Structure

```
DeepMalig-CNAx/
├── filter_cna.py           # Step 0: Filter CNA bins by variance
├── build_graph_inputs.py   # Step 1: Build graph inputs (.npz)
├── train.py                # Step 2: Train GAT-AE, produce latent + cluster labels
├── find_tumor_clusters.py  # Step 3: Unsupervised tumor cluster calling
├── evaluate.py             # Step 4: Evaluate against ground-truth labels
├── gat_conv.py             # Graph Attention Convolution layer (TensorFlow/Keras)
├── models.py               # GAT Autoencoder model definition
└── utils.py                # Graph normalization and kNN graph utilities
```

---

## Installation

```bash
conda create -n deepmalig python=3.9
conda activate deepmalig
pip install tensorflow scikit-learn pandas numpy scipy leidenalg igraph scanpy anndata
```

> `train.py` uses Leiden clustering, which requires `leidenalg` and `igraph`. Pass `--clusterer kmeans` to skip this dependency.

---

## Usage

### Step 0 — Filter CNA bins

Removes genomic bins whose variance across cells falls below a threshold. Low-variance bins carry little signal and add noise to the downstream kNN graph.

```bash
python filter_cna.py \
  --cna_csv   <path/to/CNA_matrix_raw.csv> \
  --out_csv   <path/to/CNA_matrix_filtered.csv> \
  --min_var   0.02
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--cna_csv` | str | required | Input CNA matrix CSV; first column must be `cell_name`, remaining columns are genomic bins |
| `--out_csv` | str | required | Output filtered CSV path |
| `--min_var` | float | `0.05` | Keep bins with variance ≥ this value |
| `--dtype` | str | `float32` | Numeric dtype for loading the matrix (`float32` or `float64`) |

The script prints bin variance quantiles at `[0, 10, 25, 50, 75, 90, 95, 99, 100]` percentiles for sanity-checking the threshold. 

---

### Step 1 — Build graph inputs

Builds the `.npz` bundle consumed by `train.py`. Two things happen:

1. **Node features:** UMI counts are CP10k-normalised and log1p-transformed, then subsetted to the provided signature gene list. Gene symbols are matched robustly (case-insensitive, Ensembl version suffixes stripped, surrounding quotes removed).

2. **Edges:** A kNN graph is built over cells using cosine similarity on the filtered CNA matrix. Similarities are raised to `--power` to sharpen the weight distribution, then symmetrised by taking the max weight for each undirected pair.

```bash
python build_graph_inputs.py \
  --cna_csv       <path/to/CNA_matrix_filtered.csv> \
  --cells_csv     <path/to/Cells.csv> \
  --genes_txt     <path/to/Genes.txt> \
  --mtx           <path/to/Exp_data_UMIcounts.mtx> \
  --sig_genes_txt <path/to/signature_genes.txt> \
  --k             20 \
  --power         4.0 \
  --allow_missing_cna \
  --out           <path/to/output.npz>
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--cna_csv` | str | required | Filtered CNA matrix (output of `filter_cna.py`) |
| `--cells_csv` | str | required | Cell metadata CSV; must contain a `cell_name` column |
| `--genes_txt` | str | required | Gene list corresponding to the MTX gene axis (one per line) |
| `--mtx` | str | required | UMI count matrix in Matrix Market format; accepted as either `(genes × cells)` or `(cells × genes)` |
| `--sig_genes_txt` | str | required | Signature gene symbols to use as node features (one per line) |
| `--k` | int | `20` | Number of nearest neighbours for the CNA-based kNN graph |
| `--power` | float | `4.0` | Exponent applied to cosine similarity weights; higher values produce sparser, sharper edge weights |
| `--symmetrize` / `--no_symmetrize` | flag | `True` | Whether to symmetrise the directed kNN graph |
| `--coalesce` | str | `max` | How to merge forward/reverse edge weights when symmetrising: `max` or `mean` |
| `--seed` | int | `0` | Random seed for kNN |
| `--allow_missing_cna` | flag | off | If cells in `Cells.csv` are absent from the CNA matrix, restrict to the intersection instead of raising an error |
| `--out` | str | required | Output `.npz` path |

**Output NPZ arrays:**

| Key | Shape | Description |
|---|---|---|
| `rna_feat` | `(N, G_sig)` | CP10k + log1p RNA features for signature genes |
| `edge_index` | `(2, E)` | Source/destination node indices |
| `edge_weight` | `(E,)` | CNA cosine similarity weights |
| `cell_names` | `(N,)` | Ordered cell names |
| `params` | scalar | JSON string of run parameters |

---

### Step 2 — Train

Trains the GAT Autoencoder on the graph and clusters cells in the learned latent space.

```bash
python train.py \
  --npz             <path/to/inputs.npz> \
  --hidden1         64 \
  --hidden2         16 \
  --epochs          1000 \
  --lr              1e-3 \
  --clusterer       leiden \
  --latent_knn      30 \
  --resolution      2.0 \
  --lambda_contrast 0.25 \
  --temperature     0.1 \
  --n_neg           256 \
  --zscore_features \
  --out_prefix      <path/to/output_prefix>
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--npz` | str | required | Input `.npz` from `build_graph_inputs.py` |
| `--hidden1` | int | — | Encoder layer 1 output dimension |
| `--hidden2` | int | — | Latent dimension (encoder layer 2 output) |
| `--epochs` | int | — | Number of training epochs |
| `--lr` | float | — | Adam learning rate |
| `--clusterer` | str | — | Clustering algorithm for the latent space: `leiden` or `kmeans` |
| `--latent_knn` | int | — | Neighbours for the latent-space kNN graph used by Leiden |
| `--resolution` | float | — | Leiden resolution (higher → more, smaller clusters) |
| `--lambda_contrast` | float | — | Weight of contrastive loss relative to reconstruction loss |
| `--temperature` | float | — | Temperature for InfoNCE-style contrastive loss |
| `--n_neg` | int | — | Number of negative samples per anchor in contrastive loss |
| `--zscore_features` | flag | off | Z-score normalise RNA features before training |
| `--out_prefix` | str | required | Prefix for all output files |

**Output files:**

| File | Description |
|---|---|
| `<prefix>.cells.txt` | Ordered cell names (one per line) |
| `<prefix>.labels.tsv` | Integer cluster label per cell (no header, 1 column) |
| `<prefix>.latent.tsv` | Latent coordinates, tab-separated, shape `(N, hidden2)` |

---

### Step 3 — Find tumor clusters

Scores each cluster by its median per-cell CNA burden (mean absolute deviation from the per-genomic-bin population median), then fits a **2-component 1D Gaussian Mixture Model** to separate high-burden (tumor) from low-burden (normal) clusters. Clusters whose posterior probability of belonging to the high-burden component exceeds `--tumor_posterior_thr` are called as tumor.

An optional **auto-refinement** pass splits ambiguous or large normal clusters in latent space using K-Means (with optional automatic K selection by silhouette score), then re-runs the GMM on the resulting sub-clusters. This recovers tumor sub-populations mixed into large normal clusters.

```bash
python find_tumor_clusters.py \
  --cna_csv                 <path/to/CNA_matrix_filtered.csv> \
  --names_txt               <prefix>.cells.txt \
  --labels_tsv              <prefix>.labels.tsv \
  --latent_tsv              <prefix>.latent.tsv \
  --out_prefix              <path/to/output_prefix> \
  --allow_missing_cna \
  --center_mode             per_segment_median \
  --cna_burden_mode         mean_abs \
  --cluster_stat            median \
  --log1p_cna \
  --tumor_posterior_thr     0.5 \
  --auto_refine \
  --also_refine_big_normals \
  --topk_big_normals        1 \
  --min_cluster_size_refine 800 \
  --max_refine_clusters     3 \
  --ambig_delta             0.12 \
  --k_auto --k_min 2 --k_max 10
```

**CNA burden arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--center_mode` | str | `per_segment_median` | Center CNA profiles before computing burden: `per_segment_median`, `per_segment_mean`, or `none` |
| `--cna_burden_mode` | str | `mean_abs` | Per-cell burden summary: `mean_abs` (mean absolute deviation) or `mean_sq` |
| `--cluster_stat` | str | `median` | Aggregate per-cell burdens per cluster: `median` or `mean` |
| `--log1p_cna` | flag | off | Apply log1p to per-cell burden values before GMM fitting |
| `--tumor_posterior_thr` | float | `0.5` | Minimum GMM posterior probability to call a cluster as tumor |

**Auto-refinement arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--auto_refine` | flag | off | Enable automatic refinement of ambiguous/large clusters |
| `--ambig_delta` | float | `0.12` | Clusters with posterior in `[0.5 ± delta]` are split |
| `--also_refine_big_normals` | flag | off | Also split the largest predicted-normal clusters even if not ambiguous |
| `--topk_big_normals` | int | `1` | Number of largest normal clusters to refine |
| `--min_cluster_size_refine` | int | `500` | Minimum cluster size to be eligible for refinement |
| `--max_refine_clusters` | int | `3` | Maximum total number of clusters to refine |
| `--k_auto` | flag | off | Automatically choose K for splitting by silhouette score |
| `--k_min` / `--k_max` | int | `2` / `10` | Search range when `--k_auto` is set |
| `--k_fixed` | int | `2` | Fixed K when `--k_auto` is not set |
| `--allow_missing_cna` | flag | off | Allow cells absent from the CNA matrix (uses intersection) |
| `--seed` | int | `42` | Random seed |

**Output files:**

| File | Description |
|---|---|
| `<prefix>.labels.tsv` | Final cluster labels after any refinement (1 column, no header) |
| `<prefix>.tumor_clusters.txt` | Cluster IDs called as tumor (one per line) |
| `<prefix>.cluster_scores.tsv` | Per-cluster: CNA burden score, size, GMM tumor posterior |
| `<prefix>.per_cell.tsv` | Per-cell: cell name, final cluster ID, CNA burden |
| `<prefix>.meta.json` | Full run metadata: GMM parameters, refinement targets, all arguments |

---

### Step 4 — Evaluate

Compares predicted tumor/normal assignments to ground-truth cell type labels. This step is for benchmarking only — ground truth is never used upstream.

```bash
python evaluate.py \
  --cells_csv           <path/to/Cells.csv> \
  --names_txt           <prefix>.cells.txt \
  --labels_tsv          <prefix>.labels.tsv \
  --tumor_clusters_txt  <prefix>.tumor_clusters.txt \
  --out_prefix          <path/to/output_prefix>
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--cells_csv` | str | required | Cell metadata CSV with ground-truth labels |
| `--names_txt` | str | required | Ordered cell names from training |
| `--labels_tsv` | str | required | Cluster labels (output of `find_tumor_clusters.py`) |
| `--tumor_clusters_txt` | str | required | Tumor cluster list (output of `find_tumor_clusters.py`) |
| `--out_prefix` | str | required | Prefix for evaluation outputs |
| `--cell_type_col` | str | `cell_type` | Column in `Cells.csv` containing cell type labels |
| `--malignant_key` | str | `malignant` | Value in `cell_type_col` that denotes tumor cells (case-insensitive) |

**Output files:**

| File | Description |
|---|---|
| `<prefix>.unsup_cluster_tumor_eval.summary.txt` | TP/FP/FN/TN counts, accuracy, precision, recall, F1 |
| `<prefix>.unsup_cluster_tumor_eval.per_cell.tsv` | Per-cell ground truth, cluster ID, and predicted tumor label |

---

## Data Layout

```
datasets/
└── <dataset_name>/
    ├── Cells.csv                        # cell_name, cell_type, ...
    ├── Genes.txt                        # one gene symbol per line
    ├── Exp_data_UMIcounts.mtx           # (genes × cells) or (cells × genes)
    └── CNA_matrix/
        ├── CNA_matrix_raw.csv           # cell_name + one column per genomic bin
        └── CNA_matrix_filtered.csv      # produced by filter_cna.py

signatures/
└── master_signature_genes_unique.txt    # one gene symbol per line

results/
    └── <dataset>/
        ├── inputs/
        │   └── <run_tag>.npz
        └── runs/
            └── <run_tag>/
                ├── <run_tag>.cells.txt
                ├── <run_tag>.labels.tsv
                ├── <run_tag>.latent.tsv
                ├── <run_tag>.labels.tsv
                ├── <run_tag>.tumor_clusters.txt
                ├── <run_tag>.cluster_scores.tsv
                ├── <run_tag>.per_cell.tsv
                └── <run_tag>.meta.json
```

**`Cells.csv` required columns:** `cell_name`, and a cell type column (default `cell_type` with value `malignant` for tumor cells).

**`Genes.txt`:** The MTX gene axis. Symbols may have surrounding quotes or Ensembl version suffixes — these are stripped automatically by `build_graph_inputs.py`.


## Dependencies

| Package | Purpose |
|---|---|
| `tensorflow` | GAT model training |
| `scikit-learn` | kNN graph, GMM, KMeans, silhouette scoring |
| `scanpy` / `leidenalg` / `igraph` | Leiden clustering in `train.py` |
| `scipy` | Sparse matrix I/O (MTX format), graph normalisation |
| `pandas` / `numpy` | Data loading and manipulation |

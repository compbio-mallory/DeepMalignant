#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


def read_lines(path: str) -> list[str]:
    with open(path, "r") as f:
        return [x.strip() for x in f if x.strip()]


def load_truth(cells_csv: str, cell_type_col: str = "cell_type", malignant_key: str = "malignant") -> pd.DataFrame:
    cells = pd.read_csv(cells_csv)
    if "cell_name" not in cells.columns:
        raise ValueError("Cells.csv must contain column 'cell_name'")

    if cell_type_col not in cells.columns:
        raise ValueError(f"Cells.csv missing '{cell_type_col}'. Found: {list(cells.columns)[:20]}")

    ct = cells[cell_type_col].astype(str).str.lower()
    cells["true_is_tumor"] = (ct == malignant_key.lower())
    return cells[["cell_name", "true_is_tumor"]].copy()


def load_labels(labels_tsv: str, names: pd.Series) -> pd.Series:
    # same robust parsing as before: 1-col or 2-col
    try:
        df = pd.read_csv(labels_tsv, sep="\t", header=0, dtype=str)
        if df.shape[1] == 1 and df.columns[0] not in ("label", "cluster", "cell_name"):
            raise ValueError("likely headerless")
    except Exception:
        df = pd.read_csv(labels_tsv, sep="\t", header=None, dtype=str)

    df = df.replace("", np.nan).dropna(how="any").reset_index(drop=True)

    if df.shape[1] == 1:
        lab = df.iloc[:, 0].astype(float).astype(int).reset_index(drop=True)
        m = min(len(lab), len(names))
        if len(lab) != len(names):
            print(f"[WARN] labels({len(lab)}) != names({len(names)}); truncating to {m}")
        return lab.iloc[:m].reset_index(drop=True)

    df = df.iloc[:, :2].copy()
    df.columns = ["cell_name", "label"]
    df["cell_name"] = df["cell_name"].astype(str)
    df["label"] = df["label"].astype(float).astype(int)

    merged = pd.DataFrame({"cell_name": names.values}).merge(df, on="cell_name", how="left")
    miss = merged["label"].isna().sum()
    if miss > 0:
        ex = merged.loc[merged["label"].isna(), "cell_name"].head(5).tolist()
        raise ValueError(f"Missing labels for {miss} cells after join. Example: {ex}")
    return merged["label"].astype(int)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    acc = (tp + tn) / max(1, tp + fp + fn + tn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "acc": acc, "precision": prec, "recall": rec, "f1": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells_csv", required=True)
    ap.add_argument("--names_txt", required=True)
    ap.add_argument("--labels_tsv", required=True, help="cluster labels from training (PFX.labels.tsv)")
    ap.add_argument("--tumor_clusters_txt", required=True, help="output tumor cluster list from oneclass script")
    ap.add_argument("--out_prefix", required=True)

    ap.add_argument("--cell_type_col", default="cell_type")
    ap.add_argument("--malignant_key", default="malignant")
    args = ap.parse_args()

    names = pd.Series(read_lines(args.names_txt), dtype=str, name="cell_name")
    labs = load_labels(args.labels_tsv, names)
    if len(labs) != len(names):
        m = min(len(labs), len(names))
        names = names.iloc[:m].reset_index(drop=True)
        labs = labs.iloc[:m].reset_index(drop=True)

    tumor_clusters = set(int(x) for x in read_lines(args.tumor_clusters_txt))
    y_pred = np.array([1 if int(c) in tumor_clusters else 0 for c in labs.values], dtype=int)

    truth = load_truth(args.cells_csv, cell_type_col=args.cell_type_col, malignant_key=args.malignant_key)

    merged = pd.DataFrame({"cell_name": names.values}).merge(truth, on="cell_name", how="left")
    keep = ~merged["true_is_tumor"].isna()
    if int((~keep).sum()) > 0:
        print(f"[WARN] Dropping {(~keep).sum()} cells with missing truth in Cells.csv")
    merged = merged.loc[keep].reset_index(drop=True)
    y_true = merged["true_is_tumor"].astype(int).to_numpy()
    y_pred = y_pred[keep.to_numpy()]

    metrics = compute_metrics(y_true, y_pred)

    out_summary = f"{args.out_prefix}.unsup_cluster_tumor_eval.summary.txt"
    out_per_cell = f"{args.out_prefix}.unsup_cluster_tumor_eval.per_cell.tsv"

    with open(out_summary, "w") as f:
        f.write("=== Unsupervised tumor cluster evaluation (uses GT only for benchmarking) ===\n")
        f.write(f"cells_total_with_truth = {len(y_true)}\n")
        f.write(f"tumor_clusters = {sorted(list(tumor_clusters))}\n")
        f.write(f"TP FP FN TN = {metrics['TP']} {metrics['FP']} {metrics['FN']} {metrics['TN']}\n")
        f.write(f"acc={metrics['acc']:.4f} precision={metrics['precision']:.4f} "
                f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}\n")

    per_cell = merged.copy()
    per_cell["cluster"] = labs.values[keep.to_numpy()]
    per_cell["pred_is_tumor"] = y_pred
    per_cell.to_csv(out_per_cell, sep="\t", index=False)

    print("=== Unsupervised tumor cluster evaluation ===")
    print("tumor_clusters =", sorted(list(tumor_clusters)))
    print("TP FP FN TN =", metrics["TP"], metrics["FP"], metrics["FN"], metrics["TN"])
    print(f"acc={metrics['acc']:.4f} precision={metrics['precision']:.4f} "
          f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}")
    print("[OK] wrote:")
    print(" ", out_summary)
    print(" ", out_per_cell)


if __name__ == "__main__":
    main()
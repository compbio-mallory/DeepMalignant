#!/usr/bin/env python3
"""
Filter CNA bins (columns) by variability.

Input CNA CSV format:
  cell_name, bin1, bin2, ..., bin5000

We keep bins whose variance across cells >= --min_var
(Variance computed on raw bin values.)

Also prints quantiles of variance for sanity.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cna_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--min_var", type=float, default=0.05, help="Keep bins with variance >= this.")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    args = ap.parse_args()

    cna = pd.read_csv(args.cna_csv)
    if "cell_name" not in cna.columns:
        raise ValueError("Expected a 'cell_name' column")

    cell_name = cna["cell_name"].astype(str)
    X = cna.drop(columns=["cell_name"]).to_numpy(dtype=getattr(np, args.dtype), copy=False)

    var = X.var(axis=0)
    keep = var >= args.min_var

    kept = int(keep.sum())
    dropped = int((~keep).sum())
    print(f"[IN ] cells: {X.shape[0]} bins: {X.shape[1]}")
    print(f"[OUT] bins kept: {kept} bins dropped: {dropped}")
    qs = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    print("var quantiles:", {q: float(np.percentile(var, q)) for q in qs})

    out = pd.DataFrame(X[:, keep], columns=np.array(cna.columns[1:])[keep])
    out.insert(0, "cell_name", cell_name.values)
    out.to_csv(args.out_csv, index=False)
    print("[OK] wrote:", args.out_csv)

if __name__ == "__main__":
    main()

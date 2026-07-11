"""Compare the OOS cost DISTRIBUTION of a lab-produced trajectory CSV against the
frozen hydrothermal-mpc CSV, for one policy (SDDP-IH or CVaR-SDDP-IH).

This is the "baseline shift" quantifier for MIGRATION-mpc-vs-sddp.md step 2. Both CSVs
share the per-(realisation, stage) schema (op_cost, penalty_cost, se_i, gh_i, sp_i,
thermal_i, deficit_i, inflow_i). Per-realisation total OOS cost is the undiscounted sum of
stage_objective == (op_cost + penalty_cost) over the 60 stages, matching how both files were
built (_stage_row / export_stage_trajectories). We report mean, std, CVaR95 of that total,
plus per-stage mean thermal / storage / deficit bands so a structural divergence would show.

    python validate/compare_oos_dist.py FROZEN_CSV LAB_CSV [policy_name]
"""

from __future__ import annotations
import sys
import numpy as np
import pandas as pd


def cvar(x: np.ndarray, beta: float = 0.95) -> float:
    """Mean of the worst (1-beta) tail of costs (upper tail, since cost is bad)."""
    s = np.sort(x)
    k = int(np.ceil(beta * len(s)))
    return float(s[k - 1 :].mean())


def ree_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [
        c
        for c in df.columns
        if c.startswith(prefix + "_") and c[len(prefix) + 1 :].isdigit()
    ]
    return sorted(cols, key=lambda c: int(c.split("_")[1]))


def total_cost_per_real(df: pd.DataFrame) -> np.ndarray:
    df = df.copy()
    df["stage_cost"] = df["op_cost"] + df["penalty_cost"]
    g = df.groupby("realisation")["stage_cost"].sum()
    return g.values


def per_stage_band(df: pd.DataFrame, prefix: str) -> pd.Series:
    cols = ree_cols(df, prefix)
    df = df.copy()
    df["_tot"] = df[cols].sum(axis=1)
    return df.groupby("stage")["_tot"].mean()


def summarize(df: pd.DataFrame) -> dict:
    tc = total_cost_per_real(df)
    return dict(
        n=len(tc),
        mean=tc.mean(),
        std=tc.std(ddof=1),
        cvar95=cvar(tc),
        p50=np.median(tc),
        tc=tc,
    )


def pct(a, b):
    return 100.0 * (a - b) / abs(b) if b else float("nan")


def main():
    frozen_csv, lab_csv = sys.argv[1], sys.argv[2]
    policy = sys.argv[3] if len(sys.argv) > 3 else None

    fz = pd.read_csv(frozen_csv)
    lb = pd.read_csv(lab_csv)
    if policy:
        fz = fz[fz["policy"] == policy]
        lb = lb[lb["policy"] == policy]

    F, L = summarize(fz), summarize(lb)
    print(f"=== OOS cost distribution: FROZEN vs LAB  (policy={policy}) ===")
    print(f"  frozen: {frozen_csv}")
    print(f"  lab   : {lab_csv}")
    print(f"  N realisations   frozen={F['n']}  lab={L['n']}")
    print()
    print(f"{'metric':<14}{'frozen':>16}{'lab':>16}{'delta':>16}{'delta %':>10}")
    for key, label in [
        ("mean", "mean"),
        ("p50", "median"),
        ("std", "std"),
        ("cvar95", "CVaR95"),
    ]:
        d = L[key] - F[key]
        print(
            f"{label:<14}{F[key]:>16,.0f}{L[key]:>16,.0f}{d:>16,.0f}{pct(L[key], F[key]):>9.2f}%"
        )

    # Shift of the MEAN expressed in units of the per-realisation std and of the
    # standard error of the mean (the sampling-noise yardstick).
    sem_frozen = F["std"] / np.sqrt(F["n"])
    print()
    print(
        f"mean shift  = {pct(L['mean'], F['mean']):+.2f}%  "
        f"= {(L['mean'] - F['mean']) / F['std']:+.3f} realisation-std  "
        f"= {(L['mean'] - F['mean']) / sem_frozen:+.1f} SEM"
    )
    print(
        f"SAA yardstick 1/sqrt(20) = {1 / np.sqrt(20) * 100:.1f}% (expected order of bank-swap noise)"
    )

    # Per-stage bands: max relative divergence across stages (structural check).
    print()
    print(
        f"{'per-stage band':<16}{'max|Δ%| stage':>16}{'frozen@that':>16}{'lab@that':>16}"
    )
    for prefix, label in [
        ("thermal", "thermal"),
        ("se", "storage"),
        ("deficit", "deficit"),
    ]:
        bf, bl = per_stage_band(fz, prefix), per_stage_band(lb, prefix)
        idx = bf.index.intersection(bl.index)
        bf, bl = bf.loc[idx], bl.loc[idx]
        rel = (bl - bf) / bf.replace(0, np.nan).abs()
        if rel.dropna().empty:
            worst = idx[(bl - bf).abs().values.argmax()]
        else:
            worst = rel.abs().idxmax()
        print(
            f"{label:<16}{(rel.loc[worst] * 100 if worst in rel.index and pd.notna(rel.loc[worst]) else float('nan')):>15.2f}%"
            f"{bf.loc[worst]:>16,.0f}{bl.loc[worst]:>16,.0f}  (stage {worst})"
        )


if __name__ == "__main__":
    main()

"""Aggregate a 4-res scale sweep into the decision-gate table + plot.

Reads the `results_<case>_S*_lam*.json` files emitted by run_spike across a
grid-resolution sweep and reports time-to-converge, sweeps, and value vs S —
the "find the 16 GB / minutes wall" deliverable (handoff task 3). The VRAM gap
vs SDDP is a separate scalar (validate/compare.py).

    python validate/analyze_sweep.py 'results_4ree_S*_lam0.0.json'

Plots via plotting_machine if available (the default engine); otherwise prints
the table only. Pure aggregation — safe to run anytime (no solve).
"""

from __future__ import annotations

import glob
import json
import sys


def main(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"no files match {pattern!r}")
        return
    rows = []
    for f in files:
        d = json.loads(open(f).read())
        rows.append(d)
    rows.sort(key=lambda r: r["S"])

    hdr = f"{'S':>4} {'conv':>5} {'sweeps':>7} {'sec':>9} {'V0(s0)':>14}"
    if any("simulate" in r for r in rows):
        hdr += f" {'OOS_stage':>12}"
    print(f"=== 4-res scale sweep (case={rows[0]['case']}, lam={rows[0]['lam']}) ===")
    print(hdr)
    for r in rows:
        line = (f"{r['S']:>4} {str(r['converged']):>5} {r['sweeps']:>7} "
                f"{r['seconds']:>9.1f} {r['value_at_s0']:>14,.0f}")
        if "simulate" in r:
            line += f" {r['simulate']['mean_stage_cost']:>12,.0f}"
        print(line)

    # wall: largest S that converged within ~minutes
    ok = [r for r in rows if r["converged"] and r["seconds"] <= 600]
    if ok:
        print(f"\nlargest S converging <10min: S={max(r['S'] for r in ok)}")
    print("decision gate: cross-check 4-res gap vs SDDP via validate/compare.py")

    try:  # plot is optional; a real figure can go through plotting_machine later
        import matplotlib.pyplot as plt
        S = [r["S"] for r in rows]
        sec = [r["seconds"] for r in rows]
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(S, sec, "o-")
        ax.set_xlabel("states per reservoir S")
        ax.set_ylabel("time to converge (s)")
        ax.set_title("grid-DP 4-res: time vs grid resolution")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("sweep_time_vs_S.png", dpi=130)
        print("wrote sweep_time_vs_S.png")
    except Exception as e:  # plotting optional
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_4ree_S*_lam0.0.json")

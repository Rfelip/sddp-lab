"""Cross-check the grid-DP cost against the SDDP.jl baseline (decision gate).

The comparable scalar is the discounted expected cost from the initial storage
under the cyclic-0.9 criterion:
  * grid-DP  : V_0(s0)         -> results_*.json["value_at_s0"]
  * SDDP.jl  : trained lower bound (read from the policy log / SDDP.log)

and, as a robustness check, the out-of-sample mean stage cost of each policy
(grid-DP: results json ["simulate"]["mean_stage_cost"]; SDDP: simulation mean).

SDDP.jl writes parquet whose schema we don't hard-code here; paste the two
SDDP scalars on the CLI (read them from SDDP.log / the simulation summary):

    python validate/compare.py --griddp results_2ree_S25_lam0.0.json \
        --sddp-bound 1.83e7 --sddp-sim 1.79e7

Gate (per handoff): grid-DP within a few % of SDDP AND policy cost within a
few % -> the method is a live cross-check. Gap > 5-10% at S>=15 -> grid-DP
confirmed dead for 4-res, redirect to the block-batch lane.
"""

from __future__ import annotations

import argparse
import json


def pct(a, b):
    return 100.0 * (a - b) / abs(b) if b else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--griddp", required=True, help="grid-DP results_*.json")
    ap.add_argument("--sddp-bound", type=float, required=True,
                    help="SDDP trained lower bound (discounted, from SDDP.log)")
    ap.add_argument("--sddp-sim", type=float, default=None,
                    help="SDDP out-of-sample mean cost (optional)")
    args = ap.parse_args()

    g = json.loads(open(args.griddp).read())
    v0 = g["value_at_s0"]
    print(f"=== grid-DP vs SDDP  (case={g['case']} S={g['S']} lam={g['lam']}) ===")

    # (1) informational: grid-DP value vs SDDP LOWER BOUND. The bound is a
    # relaxation, so any actual policy (incl. SDDP's own) sits above it; the
    # gap here mixes discretization with SDDP's residual optimality gap.
    print(f"grid-DP V_0(s0)        : {v0:,.0f}")
    print(f"SDDP lower bound       : {args.sddp_bound:,.0f}")
    print(f"  value vs bound (info) : {pct(v0, args.sddp_bound):+.2f}%  (>=0 expected)")

    # (2) THE GATE: policy cost vs policy cost. Both methods yield an operating
    # policy; compare grid-DP's discounted cost-to-go from s0 against SDDP's
    # simulated discounted cost. This is apples-to-apples.
    gate = None
    if args.sddp_sim is not None and "simulate" in g:
        gdp_cost = g["simulate"]["mean_discounted"]
        gate = pct(gdp_cost, args.sddp_sim)
        print(f"grid-DP policy cost    : {gdp_cost:,.0f}  (OOS discounted)")
        print(f"SDDP   policy cost     : {args.sddp_sim:,.0f}  (sim discounted)")
        print(f"  -> POLICY GAP        : {gate:+.2f}%")

    key = gate if gate is not None else pct(v0, args.sddp_bound)
    verdict = "PASS (live cross-check)" if abs(key) <= 5 else (
        "MARGINAL (5-10%)" if abs(key) <= 10 else "FAIL (grid-DP dead here)")
    print(f"\nDECISION GATE (on policy gap): {verdict}")


if __name__ == "__main__":
    main()

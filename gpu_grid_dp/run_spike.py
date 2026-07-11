"""CLI entry for the GPU grid-DP spike.  *** This is the gated 'run' step. ***

Modes:
  --dry-run   Build the problem + print grid sizes and the VRAM estimate.
              Does NOT compile or launch any JAX kernel -> no GPU/thread
              contention. Safe to run anytime to sanity-check the instance.
  --smoke     Tiny 2-res VI on CPU (override S/P/R small) to validate the math
              end-to-end without the GPU. Set JAX_PLATFORMS=cpu.
  (default)   Full value iteration on the configured case, saves results.

Run with the ROCm JAX venv for GPU:
    ~/Desktop/experimentos/.venv-rocm/bin/python -m gpu_grid_dp.run_spike --case 2ree
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import GridDPConfig
from .problem import build_problem


def estimate_vram_gb(spec, cfg) -> float:
    sz = cfg.grid_sizes(spec.n_res)
    # dominant intermediates in chunk_kernel: a handful of [chunk,A,R,d] and
    # [chunk,A,R] float32 tensors (next_raw/next_state/spill + gather temporaries)
    chunk = min(cfg.state_chunk, sz["n_states"])
    A, R, d = sz["n_actions"], cfg.n_inflow_samples, spec.n_res
    big4 = chunk * A * R * d * 4          # [chunk,A,R,d]
    big3 = chunk * A * R * 4              # [chunk,A,R]
    # measured-calibrated: the 2^d-corner interp gather + min/cvar create more
    # live [chunk,A,R(,d)] temporaries than naive fusion suggests. S=15/P=10/
    # chunk=512 actually peaked ~19 GB vs this formula's ~11 GB, so ~2x it.
    peak = 2.0 * (3 * big4 + 6 * big3)
    v_store = cfg.n_stages_cycle * sz["n_states"] * 4 * 2  # V + policy, all months
    return (peak + v_store) / 1e9


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="2ree", choices=["2ree", "4ree"])
    ap.add_argument("--S", type=int, default=None, help="states per reservoir")
    ap.add_argument("--P", type=int, default=None, help="actions per reservoir")
    ap.add_argument("--R", type=int, default=None, help="inflow samples")
    ap.add_argument("--lam", type=float, default=None, help="CVaR weight lambda")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--simulate", action="store_true", help="OOS policy sim")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    over = {}
    if args.smoke:
        over.update(n_states_per_res=8, n_actions_per_res=5, n_inflow_samples=12,
                    vi_max_sweeps=80, state_chunk=4096)
    for k_arg, k_cfg in [("S", "n_states_per_res"), ("P", "n_actions_per_res"),
                         ("R", "n_inflow_samples"), ("lam", "cvar_lambda"),
                         ("alpha", "cvar_alpha"), ("chunk", "state_chunk")]:
        v = getattr(args, k_arg)
        if v is not None:
            over[k_cfg] = v
    cfg = GridDPConfig(case=args.case, **over)

    spec = build_problem(cfg.case, cfg.iid_inflows)
    sizes = cfg.grid_sizes(spec.n_res)
    vram = estimate_vram_gb(spec, cfg)

    print(f"=== grid-DP spike: case={cfg.case} d={spec.n_res} ===")
    print(f"S={cfg.n_states_per_res} P={cfg.n_actions_per_res} R={cfg.n_inflow_samples}"
          f"  lambda={cfg.cvar_lambda} alpha={cfg.cvar_alpha} gamma={cfg.discount}")
    print(f"states S^d = {sizes['n_states']:,}   actions P^d = {sizes['n_actions']:,}"
          f"   chunks = {(sizes['n_states']+cfg.state_chunk-1)//cfg.state_chunk}")
    print(f"est. peak VRAM ~ {vram:.2f} GB   (16 GB wall)")
    print(f"load_cycle (mean {spec.load_cycle.mean():.0f}): "
          f"{np.round(spec.load_cycle).astype(int)}")
    print(f"thermal cap {spec.thermal.cap.sum():.0f}  deficit ${spec.thermal.deficit_cost}")

    if args.dry_run:
        print("[dry-run] no kernel launched.")
        return

    from .value_iteration import run_value_iteration, simulate_policy, value_at
    res = run_value_iteration(spec, cfg)
    v0 = value_at(spec, cfg, res, spec.init_storage, season=0)
    print(f"\nconverged={res.converged} sweeps={res.sweeps} "
          f"time={res.seconds:.1f}s  V_0(s0)={v0:,.1f}")

    out = dict(case=cfg.case, S=cfg.n_states_per_res, P=cfg.n_actions_per_res,
               R=cfg.n_inflow_samples, lam=cfg.cvar_lambda, gamma=cfg.discount,
               converged=res.converged, sweeps=res.sweeps, seconds=res.seconds,
               value_at_s0=v0)
    if args.simulate:
        sim = simulate_policy(spec, cfg, res)
        out["simulate"] = sim
        print(f"OOS  mean_stage_cost={sim['mean_stage_cost']:,.1f}  "
              f"mean_discounted={sim['mean_discounted']:,.1f}")

    out_path = Path(args.out or f"results_{cfg.case}_S{cfg.n_states_per_res}"
                    f"_lam{cfg.cvar_lambda}.json")
    out_path.write_text(json.dumps(out, indent=2))
    np.savez(out_path.with_suffix(".npz"), V=res.V, history=np.array(res.history))
    print(f"saved {out_path} (+ .npz)")


if __name__ == "__main__":
    main()

"""Configuration for the GPU grid-DP hydrothermal spike.

A feasibility spike (Lee & Sun 2025, arXiv:2511.15629) generalized from 1-D
storage arbitrage to a d-reservoir hydrothermal value-function DP, with:
  * cyclic (infinite-horizon, discounted) value iteration to a fixed point,
  * mean-CVaR risk reduction over inflow samples,
  * NO LP solver: the stage cost is a precomputed 1-D merit-order thermal
    lookup over total hydro energy.

See gpu_grid_dp/README.md and GPU_GRID_DP_HANDOFF.md for the plan and the
decision gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GridDPConfig:
    # --- discretization ---------------------------------------------------
    n_states_per_res: int = 25          # S: storage levels per reservoir
    n_actions_per_res: int = 11         # P: turbined-release levels per reservoir
    n_inflow_samples: int = 30          # R: openings per stage

    # --- horizon / discount ----------------------------------------------
    n_stages_cycle: int = 12            # cyclic period (months)
    discount: float = 0.9               # gamma; cyclic discounted value iteration

    # --- risk measure: rho = (1-lambda) E + lambda AVaR_alpha -------------
    # lambda=0 -> risk-neutral (pure expectation), matches a risk-neutral SDDP run.
    # Lab/Avila convention (run-03): CVaR variant alpha=0.1, lambda=0.15.
    cvar_alpha: float = 0.1             # AVaR tail probability
    cvar_lambda: float = 0.0            # risk weight; 0 = risk-neutral

    # --- value-iteration stopping ----------------------------------------
    vi_tol: float = 1e-3                # ||V_{k+1}-V_k||_inf relative stop
    vi_max_sweeps: int = 400            # safety cap (~65 expected per handoff)

    # --- compute ----------------------------------------------------------
    # Chunk the flattened state axis so the [chunk, P^d, R] work tensor fits
    # VRAM. NEVER materialise the full S^d x P^d x R tensor (~320 GB at d=4).
    state_chunk: int = 4096
    dtype: str = "float32"
    seed: int = 0                       # inflow sampling seed (reproducible)

    # --- problem selection ------------------------------------------------
    case: str = "2ree"                  # "2ree" (toy) or "4ree" (scale)
    iid_inflows: bool = True            # True: one inflow law for all months
                                        # (toy); False: per-month monthly laws.

    def grid_sizes(self, n_res: int) -> dict:
        S = self.n_states_per_res
        P = self.n_actions_per_res
        R = self.n_inflow_samples
        return {
            "n_res": n_res,
            "n_states": S ** n_res,
            "n_actions": P ** n_res,
            "n_samples": R,
            "work_tensor_per_chunk": self.state_chunk * (P ** n_res) * R,
        }

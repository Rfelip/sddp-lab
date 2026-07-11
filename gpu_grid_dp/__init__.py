"""GPU grid-DP hydrothermal spike (Lee & Sun 2025 generalized to d reservoirs).

LP-free, GPU-friendly discretized value-function dynamic programming as a
cross-check / alternative to SDDP for the 4-reservoir hydrothermal problem.
See README.md and ../GPU_GRID_DP_HANDOFF.md.
"""

from .config import GridDPConfig
from .problem import ProblemSpec, build_problem

__all__ = ["GridDPConfig", "ProblemSpec", "build_problem"]

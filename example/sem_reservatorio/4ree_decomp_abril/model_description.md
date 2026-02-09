# 4REE DECOMP Model Definition

## Overview
This model is a variation of the `4ree` example, adapted to follow the **DECOMP** short-term operation planning methodology. It represents a system with 4 reservoirs (hydro plants) and thermal generation.

The planning horizon is split into two phases:
1.  **Deterministic Weeks (Stages 1-4):** The first month is discretized into 4 weekly stages. Inflows are deterministic (based on the first month's forecast).
2.  **Stochastic Months (Stages 5-15):** The subsequent 11 months are modeled as monthly stages with stochastic inflows, following a scenario tree.

## Methodology

### Horizon Discretization
*   **Total Stages:** 15
*   **Stage 1:** Week 1 of Month 1 (Deterministic)
*   **Stage 2:** Week 2 of Month 1 (Deterministic)
*   **Stage 3:** Week 3 of Month 1 (Deterministic)
*   **Stage 4:** Remainder of Month 1 (Deterministic)
*   **Stages 5-15:** Months 2 to 12 (Stochastic)

### Uncertainty Structure
*   **Weeks 1-4:** Branching factor = 1. Inflows are effectively deterministic (LogNormal with negligible variance).
*   **Months 2-12:** Branching factor = 10. Inflows follow the standard stochastic process defined for the system.

### Mathematical Formulation
The objective is to minimize the expected operating cost over the entire horizon:

$$ 
\min \mathbb{E} \left[ \sum_{t=1}^{15} \beta^{t-1} (c_g \cdot g_t + c_\delta \cdot \delta_t) + \alpha_{future}(v_{final}) \right] 
$$ 

(Note: In this specific implementation, the "Future Cost Function" is implicitly handled by the end-of-horizon boundary condition or cuts if coupled, but here we simulate the full 12-month horizon where the "tail" acts as the medium-term coupling).

## Implementation Details
*   **Scenarios:** `scenarios.jsonc` defines the branching structure `[1, 1, 1, 1, 10, ..., 10]`.
*   **Inflows:** `inflow_scenarios.jsonc` has been expanded to 15 seasons. Season 1 parameters were replicated for stages 1-4 with reduced variance.
*   **Stages:** `stages.csv` explicitly defines the start and end dates for the 4 initial weeks and the subsequent months.

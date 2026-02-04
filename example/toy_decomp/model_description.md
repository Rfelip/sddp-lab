# Toy DECOMP Model Definition

## Overview
This model represents a simplified hydrothermal dispatch problem designed to replicate the logic of the **DECOMP** model (short-term operation planning). It consists of a single reservoir (hydro plant), a thermal plant, and a load, over a 5-stage horizon.

The structure mimics the DECOMP methodology (as per the *Manual de Referência*):
*   **Stages 1-4 (Weeks):** Deterministic inflows (Zero Variance). This corresponds to the first month discretized into weeks with known forecasts.
*   **Stage 5 (Future/Coupling):** Stochastic inflows (100 branches). This represents the coupling with the medium-term model (**NEWAVE**).

## Mathematical Formulation

### Sets and Indices
*   $t \in \{1, \dots, T\}$: Time stages, where $T=5$.
*   $\\omega_t \in \Omega_t$: Random outcome at stage $t$.

### State Variables
*   $v_t$: Stored volume in the reservoir at the end of stage $t$ ($hm^3$). 
    *   $v_0$: Initial volume.

### Random Variables (Uncertainty)
The scenario tree follows the DECOMP standard:
*   **Stages 1-4 (Initial Month):** $a_t$ is treated as deterministic (known forecast).
    *   $a_t \sim \mathcal{N}(\\mu_t, 0)$.
*   **Stage 5 (Subsequent Months):** $a_t$ follows a stochastic process (Scenario Tree).
    *   $a_t \sim \mathcal{N}(\\mu_t, \sigma_t^2)$.

### Coupling with NEWAVE
The objective function at the final stage $T$ includes a **Future Cost Function (FCF)** derived from the medium-term model (NEWAVE).

$$ \\alpha_{T+1}(v_T, a_{past}) $$

Where $\\alpha_{T+1}$ is a polyhedral function (set of cuts) representing the expected future cost.
*   $v_T$: Stored volume at the end of the DECOMP horizon.
*   $a_{past}$: Past inflows (relevant for Auto-Regressive models, order $p$). 

### Decision Variables
*   $u_t$: Turbined volume at stage $t$ ($hm^3$).
*   $s_t$: Spilled volume at stage $t$ ($hm^3$).
*   $g_t$: Thermal generation at stage $t$ ($MWavg$).
*   $\\delta_t$: Deficit (unmet demand) at stage $t$ ($MWavg$).

### Parameters
*   $D_t$: Energy demand at stage $t$ ($MWavg$).
*   $\\rho$: Hydro production coefficient ($MWavg / (hm^3/stage)$).
*   $\\bar{v}$: Maximum reservoir capacity ($hm^3$).
*   $\\bar{u}$: Maximum turbine capacity ($hm^3$).
*   $\\bar{g}$: Maximum thermal capacity ($MWavg$).
*   $c_g$: Operating cost of thermal generation ($\\/MWh$). 
*   $c_\\delta$: Deficit cost (penalty) ($\\/MWh$).
*   $\\beta$: Discount factor.

### Objective Function
Minimize the expected total operating cost plus the future cost:

$$ 
\min \mathbb{E} \left[ \sum_{t=1}^{T} \\beta^{t-1} (c_g \cdot g_t + c_\\delta \cdot \\delta_t) + \\beta^T \cdot \\alpha_{T+1}(v_T) \right] 
$$ 

### Constraints

1.  **Water Balance (Hydro Dynamics):**
    $$v_t = v_{t-1} + a_t(\\omega_t) - u_t - s_t, \quad \forall t$$ 

2.  **Energy Balance (Demand Satisfaction):**
    $$\\rho \cdot u_t + g_t + \\delta_t = D_t, \quad \forall t$$ 

3.  **Bounds:**
    $$0 \le v_t \le \\bar{v}$$ 
    $$0 \le u_t \le \\bar{u}$$ 
    $$0 \le s_t$$ 
    $$0 \le g_t \le \\bar{g}$$ 
    $$0 \le \\delta_t$$ 

## Implementation Details

### Branching Structure (`scenarios.jsonc`)
The scenario tree uses the following branching factors per stage:
*   Stage 1: 1 branch (Deterministic)
*   Stage 2: 1 branch (Deterministic)
*   Stage 3: 1 branch (Deterministic)
*   Stage 4: 1 branch (Deterministic)
*   Stage 5: 100 branches (Stochastic/Coupling)

### Distributions (`inflow_scenarios.jsonc`)
*   **Inflow ($a_t$):** Modeled using `Normal` distributions.
    *   Stages 1-4: $\\sigma = 0.0$ (Zero Variance).
    *   Stage 5: $\\sigma = 10.0$ (Stochastic).
# São Francisco cascade (`saofrancisco`)

A **real 7-plant São Francisco river cascade** (Brazilian Northeast), ported from the WSMPI-2023 study-group
paper *"Impact of environmental constraints in hydrothermal energy planning"* (Bueno, Diniz, Lobato,
Sagastizábal, Vinente, 2023; partnered with CEPEL/CCEE, benchmarked vs NEWAVE). Single-bus, cyclic
(infinite-horizon, discount 0.9, 12-month cycle).

## Topology (a real cascade — water flows top → bottom)
`Retiro Baixo → Três Marias → Sobradinho → Itaparica → Comp PAF-MOX → Xingó → outlet`, with
`Queimado → Sobradinho` (a confluence). Plants 6–7 are run-of-river (zero storage). Decision-complex:
upstream water generates at every downstream plant.

## Units
Storage / turbine / spill in **hm³**; generation in **MWmonth** (`gen = productivity·turbined`); deficit
cost 7643.82 R$/MWh. Inflow data (WSMPI `CenariosAfluencias.csv`, 100 real series, m³/s) was converted to
hm³ (×K=2.628 = 730 h · 3600 s / 10⁶) and fitted to a per-(plant,month) **LogNormal** (zeros floored;
Xingó ≈ 0, run-of-river). Identity copula (independent subsystems) — a fitted-Σ copula is a follow-up.

## Build / regenerate
`python3 build_saofrancisco.py` (reads `~/Desktop/wsmpi-2023/Dados/`). Verified: trains under HiGHS, storage
stays ≥ 0.

## Not yet modeled
The paper's **environmental constraints** (ANA state-dependent min/max-outflow rules; the
`RestricoesSeguranca_*.csv` tables) are omitted here — this is a clean cascade. Add them later as a Variant
(0-1 / SDDiP or piecewise-linear). Full provenance + data tables:
`personal_obsidian/Virgil/notes/saofrancisco-cascade-model.md`.

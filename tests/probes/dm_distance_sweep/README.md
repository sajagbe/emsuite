# Dipole distance probe

## Question

Does emsuite’s `dm` report a **tuning effect** (MM charge polarizes the QM
wavefunction → density-derived dipole changes), or does it also **include the
MM charge itself** in the dipole operator (μ += Σ qᵢ rᵢ)?

Those are different:

| Mechanism | Distance dependence (charge moved farther out) |
|-----------|--------------------------------------------------|
| Tuning only (μ from QM density + nuclei) | \|Δμ\| should **shrink** toward 0 (field ~ 1/r²) |
| MM charge included in μ | \|μ\| / direct contribution often **grows** with \|r\| |

## What emsuite calls

```python
results["dm"] = float(np.linalg.norm(mf.dip_moment()))
```

PySCF `scf.hf.dip_moment` is

```
μ = Σ_A Q_A R_A − Tr(P · r)
```

using **only QM atoms** for the nuclear term. MM charges enter SCF via
`hcore` / `energy_nuc` but are **not** added into `dip_moment`. So the code
path *should* be tuning-only — this experiment checks that empirically.

## Protocol

1. Pick one VDW surface point with a large \|`dm_effect`\| (water idx 15 /
   MAX, and idx 10 / MIN).
2. Vacuum B3LYP/6-31G\* → baseline \|μ\|.
3. Place q = +0.10 along the COM→point ray at outward offsets
   δ = 0, 1.0, 1.1, …, 3.0 Å from the native surface position.
4. Re-SCF with that single MM charge; record \|μ\| and Δμ = \|μ\| − \|μ₀\|.
5. Also record the counterfactual μ_incl = μ_QM + q·(r − origin) to show what
   “charge included in μ” would look like.

## Run

```bash
cd /data/PHO_WORK/sajagbe2/packages/emsuite/.worktrees/dm-distance-sweep
uv run python tests/probes/dm_distance_sweep/run_dm_distance_sweep.py
```

Outputs land in `tests/probes/dm_distance_sweep/out/`.

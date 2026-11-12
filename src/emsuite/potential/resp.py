"""RESP partial charges (HF/6-31G*) for ligand PQR files.

NumPy/PySCF and CuPy/GPU4PySCF implementations derived from
``gpu4pyscf.pop.esp`` (gpu4pyscf 1.4.3), which is
Copyright 2021-2024 The PySCF Developers, Apache License 2.0; reference cited
there: https://doi.org/10.1002/qua.26035. Validated against the GPU module in
tests/probes/resp_cpu/ (same density: charges agree to 1e-11 e).

Method: electrostatic potential sampled on Merz-Kollman shells at 1.4, 1.6, 1.8
and 2.0 x the atomic radii, least-squares fit of atom charges with the total
charge fixed, then two-stage RESP with graph equivalence and stage-2 fixed atoms.
Automatic execution prefers one usable GPU; unavailable CUDA or GPU4PySCF falls
back to CPU before the calculation. Numerical failures are reported, not retried
silently on another backend.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from time import perf_counter

import numpy as np
from pyscf import gto, scf
from pyscf.data import radii
from scipy.spatial import distance_matrix

from emsuite.geometry import read_xyz
from emsuite.physconst import vdw_radius_angstrom

# Merz-Kollman radii (Å) from GAMESS, as used by gpu4pyscf; H..Cl only.
_MK_RADII_A = {
    "H": 1.20, "He": 1.20, "Li": 1.37, "Be": 1.45, "B": 1.45, "C": 1.50, "N": 1.50,
    "O": 1.40, "F": 1.35, "Ne": 1.30, "Na": 1.57, "Mg": 1.36, "Al": 1.24, "Si": 1.17,
    "P": 1.80, "S": 1.75, "Cl": 1.70,
}  # fmt: skip

DEFAULT_BASIS = "6-31g*"
_GRID_DENSITY = 1.0 * radii.BOHR**2  # points per Bohr^2 (gpu4pyscf default)
_SHELL_SCALES = (1.4, 1.6, 1.8, 2.0)


def _shell_radius_bohr(symbol: str) -> float:
    """Merz-Kollman radius for H..Cl; the emsuite VDW table for heavier elements."""
    r = _MK_RADII_A.get(symbol)
    if r is None:
        r = vdw_radius_angstrom(symbol)
    return r / radii.BOHR


def _unit_surface(n: int) -> np.ndarray:
    """Points on the unit sphere (at most n). Same construction as gpu4pyscf."""
    ux, uy, uz = [], [], []
    eps = 1e-10
    nequat = int(np.sqrt(np.pi * n))
    nvert = int(nequat / 2)
    for i in range(nvert + 1):
        fi = np.pi * i / nvert
        z = np.cos(fi)
        xy = np.sin(fi)
        nhor = max(int(nequat * xy + eps), 1)
        fj = 2.0 * np.pi * np.arange(nhor) / nhor
        ux.append(np.cos(fj) * xy)
        uy.append(np.sin(fj) * xy)
        uz.append(z * np.ones(nhor))
    ux, uy, uz = np.concatenate(ux), np.concatenate(uy), np.concatenate(uz)
    return np.array([ux[:n], uy[:n], uz[:n]]).T


def fitting_points(mol, scales=_SHELL_SCALES, density=_GRID_DENSITY) -> np.ndarray:
    """ESP sample points (Bohr) on scaled atomic spheres, outside every other sphere."""
    coords = mol.atom_coords(unit="B")
    atom_radii = np.array([_shell_radius_bohr(s) for s in mol.elements])
    out = []
    for scale in scales:
        scaled = atom_radii * scale
        for i, coord in enumerate(coords):
            r = scaled[i]
            points = coord + r * _unit_surface(int(density * 4.0 * np.pi * r**2))
            dist = distance_matrix(points, coords) + 1e-10
            out.append(points[np.all(dist >= scaled, axis=1)])
    return np.concatenate(out)


def _electron_potential(mol, dm, points, blksize=256) -> np.ndarray:
    """sum_uv D_uv (u|1/|r-C||v) at each point C (Bohr), in blocks to bound memory."""
    dm = np.asarray(dm)
    if dm.ndim == 3:  # unrestricted: total density
        dm = dm[0] + dm[1]
    out = np.empty(len(points))
    for p0 in range(0, len(points), blksize):
        blk = points[p0 : p0 + blksize]
        out[p0 : p0 + len(blk)] = np.einsum("pij,ij->p", mol.intor("int1e_grids", grids=blk), dm)
    return out


def esp_data(mol, dm) -> tuple[np.ndarray, np.ndarray]:
    """Fitting points (Bohr) and the QM electrostatic potential there (Hartree/e)."""
    points = fitting_points(mol)
    rinv = 1.0 / distance_matrix(mol.atom_coords(unit="B"), points)
    return points, mol.atom_charges() @ rinv - _electron_potential(mol, dm, points)


def _normal_equations(mol, points, v_esp) -> tuple[np.ndarray, np.ndarray]:
    rinv = 1.0 / distance_matrix(mol.atom_coords(unit="B"), points)
    return rinv @ rinv.T, rinv @ v_esp


def _constrained_fit(
    G,
    g,
    total,
    restrained,
    resp_a,
    resp_b=0.1,
    equal=(),
    fixed=None,
    maxit=25,
    tol=1e-5,
    *,
    xp=np,
) -> np.ndarray:
    """Minimise |V - Rq|^2 + resp_a * sum_restrained (sqrt(q^2 + b^2) - b) subject to
    sum(q) = total, q_i = q_j for (i, j) in ``equal``, and q_i = v for i, v in ``fixed``.
    Lagrange multipliers; the hyperbolic restraint is iterated as in gpu4pyscf/Amber."""
    n = len(g)
    fixed = fixed or {}
    rows, rhs = [np.ones(n)], [float(total)]
    for i, j in equal:
        r = np.zeros(n)
        r[i], r[j] = 1.0, -1.0
        rows.append(r)
        rhs.append(0.0)
    for i, v in fixed.items():
        r = np.zeros(n)
        r[i] = 1.0
        rows.append(r)
        rhs.append(float(v))
    C = xp.asarray(np.array(rows))
    K0 = xp.zeros((n + len(rows), n + len(rows)))
    K0[:n, :n] = G
    K0[:n, n:] = C.T
    K0[n:, :n] = C
    b = xp.concatenate([g, xp.asarray(rhs)])
    mask = np.asarray(restrained, dtype=bool).copy()
    mask[list(fixed)] = False
    mask = xp.asarray(mask)
    q = xp.linalg.lstsq(K0, b, rcond=None)[0][:n]
    for _ in range(maxit):
        K = K0.copy()
        K[xp.arange(n), xp.arange(n)] += mask * resp_a / xp.sqrt(q**2 + resp_b**2)
        q_new = xp.linalg.lstsq(K, b, rcond=None)[0][:n]
        if float(xp.linalg.norm(q_new - q)) < tol:
            return q_new
        q = q_new
    return q


def _heavy(mol) -> np.ndarray:
    return np.array([gto.charge(s) > 1 for s in mol.elements])


def resp_fit(mol, dm, resp_a=5e-4, resp_b=0.1, maxit=25, tol=1e-5) -> np.ndarray:
    """Single-stage RESP (gpu4pyscf.pop.esp.resp_solve): restraint on heavy atoms, no
    equivalencing."""
    G, g = _normal_equations(mol, *esp_data(mol, dm))
    return _constrained_fit(G, g, mol.charge, _heavy(mol), resp_a, resp_b, maxit=maxit, tol=tol)


def equivalence_groups(mol) -> tuple[list[list[int]], list[list[int]], list[int]]:
    """Atom groups for two-stage RESP, following Amber respgen.

    Returns (stage1_groups, stage2_groups, stage2_atoms). Each group is a list of
    atom indices that must share one charge; stage2_atoms are the atoms refitted
    in stage 2 (all others keep their stage-1 charge). Symmetry comes from the bond graph (RDKit
    canonical ranks without tie breaking). CH3/CH2 groups = sp3 carbon (4
    neighbours) with >= 2 hydrogens.
      stage 1: symmetric atoms tied, except CH3/CH2 carbons and their hydrogens
      stage 2: CH3/CH2 carbons and hydrogens: same-carbon hydrogens tied, and
               symmetric atoms tied
    """
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    rw = Chem.RWMol()
    conf = Chem.Conformer(mol.natm)
    for i, (sym, xyz) in enumerate(
        zip(mol.elements, mol.atom_coords(unit="Angstrom"), strict=True)
    ):
        rw.AddAtom(Chem.Atom(sym))
        conf.SetAtomPosition(i, xyz.tolist())
    rw.AddConformer(conf)
    rdDetermineBonds.DetermineConnectivity(rw)
    rmol = rw.GetMol()
    rank = list(Chem.CanonicalRankAtoms(rmol, breakTies=False))

    in_ch = set()
    ch_h_of = {}
    for a in rmol.GetAtoms():
        hs = [n.GetIdx() for n in a.GetNeighbors() if n.GetAtomicNum() == 1]
        if a.GetAtomicNum() == 6 and a.GetDegree() == 4 and len(hs) >= 2:
            in_ch.add(a.GetIdx())
            in_ch.update(hs)
            ch_h_of[a.GetIdx()] = hs

    def by_rank(atoms):
        groups = {}
        for i in atoms:
            groups.setdefault(rank[i], []).append(i)
        return [sorted(v) for v in groups.values() if len(v) > 1]

    stage1 = by_rank([i for i in range(mol.natm) if i not in in_ch])
    stage2 = by_rank(sorted(in_ch))
    for hs in ch_h_of.values():  # same-carbon hydrogens, even if not graph-symmetric
        stage2.append(sorted(hs))
    return stage1, _merge(stage2), sorted(in_ch)


def _merge(groups):
    """Union overlapping groups."""
    merged = []
    for g in map(set, groups):
        for m in [m for m in merged if m & g]:
            g |= m
            merged.remove(m)
        merged.append(g)
    return [sorted(m) for m in merged]


def _pairs(groups):
    return [(g[0], j) for g in groups for j in g[1:]]


def resp_two_stage(mol, dm, a1=5e-4, a2=1e-3, resp_b=0.1, *, xp=np, timings=None):
    """Standard two-stage RESP (Bayly et al., J. Phys. Chem. 1993, 97, 10269),
    set up as Amber respgen does: stage 1 a = 0.0005, stage 2 a = 0.001, hydrogens
    unrestrained in both."""
    start = perf_counter()
    if xp is np:
        G, g = _normal_equations(mol, *esp_data(mol, dm))
    else:
        from gpu4pyscf.gto.int3c1e import int1e_grids

        points = fitting_points(mol)
        coords, points_gpu = xp.asarray(mol.atom_coords()), xp.asarray(points)
        rinv = 1 / xp.linalg.norm(coords[:, None] - points_gpu[None], axis=2)
        density = xp.asarray(dm)
        if density.ndim == 3:
            density = density[0] + density[1]
        potential = xp.asarray(mol.atom_charges()) @ rinv - int1e_grids(
            mol, points, dm=density, direct_scf_tol=1e-14
        )
        G, g = rinv @ rinv.T, rinv @ potential
        xp.cuda.Stream.null.synchronize()
    if timings is not None:
        timings["esp_and_normal_equations_s"] = perf_counter() - start
    start = perf_counter()
    stage1, stage2, refit = equivalence_groups(mol)
    q1 = _constrained_fit(G, g, mol.charge, _heavy(mol), a1, resp_b, equal=_pairs(stage1), xp=xp)
    if not refit:  # no CH3/CH2 groups: stage 1 is final
        q = q1
    else:
        fixed = {i: float(q1[i]) for i in range(mol.natm) if i not in set(refit)}
        q = _constrained_fit(
            G, g, mol.charge, _heavy(mol), a2, resp_b, equal=_pairs(stage2), fixed=fixed, xp=xp
        )
    if xp is not np:
        xp.cuda.Stream.null.synchronize()
    if timings is not None:
        timings["two_stage_fit_s"] = perf_counter() - start
    return q


def _gpu_backend():
    """Probe availability only; callers must pin visibility before invoking this."""
    from emsuite.core import _gpu

    if not _gpu.CUPY_AVAILABLE:
        return None, "CuPy is not installed"
    try:
        cp = _gpu.cp
        if cp.cuda.runtime.getDeviceCount() != 1:
            return None, "exactly one usable visible GPU is required"
        with cp.cuda.Device(0):
            cp.zeros(1)
            cp.cuda.Stream.null.synchronize()
        from gpu4pyscf import dft as gpu_dft
        from gpu4pyscf import scf as gpu_scf
        from gpu4pyscf.gto.int3c1e import int1e_grids  # noqa: F401

        return (cp, gpu_scf, gpu_dft), None
    except Exception as exc:
        # This boundary probes CUDA/imports only, never SCF or fitting failures.
        return None, f"GPU backend unavailable: {type(exc).__name__}: {exc}"


@contextmanager
def _backend(backend):
    from pyscf import dft

    if backend not in {"auto", "cpu", "gpu"}:
        raise ValueError("RESP backend must be auto, cpu or gpu")
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        if backend == "cpu":
            selected, reason = None, "CPU explicitly requested"
        elif previous in {"", "-1"}:
            selected, reason = None, "CUDA devices disabled by CUDA_VISIBLE_DEVICES"
        else:
            # Restore the complete allocation afterward for subsequent tuning workers.
            os.environ["CUDA_VISIBLE_DEVICES"] = (previous or "0").split(",")[0].strip()
            selected, reason = _gpu_backend()
        if selected is None:
            if backend == "gpu":
                raise RuntimeError(f"GPU RESP requested but unavailable: {reason}")
            print(f"RESP backend: CPU ({reason})")
            yield np, scf, dft
        else:
            print("RESP backend: GPU (GPU4PySCF/CuPy, one visible device)")
            cp, gpu_scf, gpu_dft = selected
            with cp.cuda.Device(0):
                yield cp, gpu_scf, gpu_dft
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous


def resp_charges_from_atoms(
    atoms: list[tuple[str, float, float, float]],
    charge: int = 0,
    spin: int = 0,
    basis: str = DEFAULT_BASIS,
    label: str = "molecule",
    method: str = "hf",
    backend: str = "auto",
    diagnostics: dict | None = None,
) -> np.ndarray:
    """Two-stage RESP charges from a ``method``/``basis`` calculation at the given geometry (Å).

    ``atoms`` is a list of (element, x, y, z). ``spin`` is 2S (0 = singlet).
    ``method`` is 'hf' or a DFT functional name such as 'b3lyp'. No geometry
    optimization is done.
    """
    if not atoms:
        raise ValueError(f"No atoms given for {label}")
    mol = gto.M(
        atom=[(s, (x, y, z)) for s, x, y, z in atoms],
        basis=basis,
        charge=charge,
        spin=spin,
        unit="Angstrom",
        verbose=0,
    )
    timings = {}
    started = perf_counter()
    with _backend(backend) as (xp, scf_module, dft_module):
        selected = "cpu" if xp is np else "gpu"
        if method.lower() == "hf":
            mf = scf_module.RHF(mol) if spin == 0 else scf_module.UHF(mol)
        else:
            mf = dft_module.RKS(mol) if spin == 0 else dft_module.UKS(mol)
            mf.xc = method
        start = perf_counter()
        mf = mf.run()
        if not mf.converged:
            mf = mf.newton().run()
        if not mf.converged:
            raise RuntimeError(f"{method.upper()}/{basis} SCF did not converge for {label}")
        if xp is not np:
            xp.cuda.Stream.null.synchronize()
        timings["scf_s"] = perf_counter() - start
        q = resp_two_stage(mol, mf.make_rdm1(), xp=xp, timings=timings)
        if xp is not np:
            q = q.get()
        energy = float(mf.e_tot)
    timings["total_s"] = perf_counter() - started
    if diagnostics is not None:
        diagnostics.update(
            backend=selected,
            method=method,
            basis=basis,
            charge=charge,
            spin=spin,
            energy_hartree=energy,
            timings_seconds=timings,
        )
    print(f"Two-stage RESP charges ({method.upper()}/{basis}) for {label}: sum = {q.sum():+.4f}")
    print("RESP timings (seconds): " + ", ".join(f"{k}={v:.6f}" for k, v in timings.items()))
    return q


def resp_charges_from_xyz(
    xyz_path: str,
    charge: int = 0,
    spin: int = 0,
    basis: str = DEFAULT_BASIS,
) -> tuple[list[tuple[str, float, float, float]], np.ndarray]:
    """HF/``basis`` RESP charges for an XYZ file; same return shape as
    ``partial_charges_from_xyz``: (atoms, charges). ``spin`` is 2S (0 = singlet)."""
    atoms = read_xyz(xyz_path)
    if not atoms:
        raise ValueError(f"No atoms found in {xyz_path}")
    return atoms, resp_charges_from_atoms(atoms, charge, spin, basis, label=xyz_path)

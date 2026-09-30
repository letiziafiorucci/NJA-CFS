#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nja_susceptibility_shielding.py
================================

Compute, for a paramagnetic complex described by an XYZ geometry and a set of
point charges (simple electrostatic / point-charge crystal-field model):

  1. the magnetic susceptibility tensor (chi) of the metal ion, using the
     ligand-field / crystal-field electronic structure engine of NJA-CFS
     (https://github.com/letiziafiorucci/NJA-CFS, branch `sph_harm_y`);

  2. the nuclear (pseudocontact/dipolar) magnetic shielding tensor sigma_I for
     every other nucleus in the structure, obtained from chi in the standard
     point-dipole approximation (Kurland-McGarvey):

        sigma_I = - 1/(4*pi*r^5) * (3 r r^T - r^2 * I) . chi        (dimensionless)

     where r is the metal->nucleus vector (r = |r|), chi is SI (m^3, single-
     molecule volume susceptibility);

  3. (optional) the derivative of chi and of the pseudocontact shifts with
     respect to selected normal-mode coordinates, read from an ORCA .hess
     file, by finite-differencing steps 1-2 for structures displaced by
     +/-A along each selected mode's Cartesian eigenvector;

  4. (optional, --plot-density) a 3D plot of the point charges together with
     the ground crystal-field state's 4f electron-density shape, in the
     Sievers-like axial approximation, population-mixed over the ground
     state's |J,MJ> composition (see plot_charge_and_density()).

------------------------------------------------------------------------------
STRUCTURE / CHARGES INPUT
------------------------------------------------------------------------------
--xyz FILE
    Standard XYZ file (2 header lines: atom count, comment; then
    "label  x  y  z" per atom, in Angstrom). The paramagnetic metal ion must
    be one of the atoms (by default the first one, override with
    --metal-index, 0-based).

--charges FILE  [--charges-format {simple,orca}]  [--charge-scheme {mulliken,loewdin}]
    simple (default):
        Plain text, one entry per line, SAME ORDER as the XYZ atoms (metal
        line included, its value is ignored). Each line is either
            <charge>
        or
            <label> <charge>
        (label, if present, is only used as a sanity check against the XYZ
        file). Blank lines and lines starting with '#' are ignored.
    orca:
        A regular ORCA .out/.log file. Partial charges are read from the
        LAST "MULLIKEN ATOMIC CHARGES" (default) or "LOEWDIN ATOMIC CHARGES"
        block (--charge-scheme), matched 1:1 against the XYZ atom order.

------------------------------------------------------------------------------
VIBRATIONAL-MODE SCAN (optional)
------------------------------------------------------------------------------
--hessian FILE
    An ORCA .hess file (contains $normal_modes, $vibrational_frequencies and,
    for the mass-weighting below, $atoms). If given, in addition to the
    equilibrium chi/shielding calculation, the script samples --n-frames
    geometries along each selected mode, with frame i (i = 0..n-1) displaced
    from equilibrium by

        Q_i = amplitude * sin(2*pi*i/(n-1))          (Angstrom)

    i.e. one full sinusoidal period that starts and ends exactly at the
    equilibrium geometry (Q_0 = Q_{n-1} = 0). chi and the pseudocontact
    shifts are recomputed at every frame, giving the full chi(Q)/shift(Q)
    trajectory for that mode (useful to see anharmonic/nonlinear response,
    or to animate the mode), plus a linear coupling constant d(chi)/dQ and
    d(shift)/dQ obtained as the least-squares (through-origin) slope of the
    trajectory against Q -- more robust to numerical noise than a plain
    2-point finite difference, since it uses every sampled frame.

--modes "7,8,9"
    0-based indices (as printed in the ORCA $normal_modes columns) of the
    modes to scan. If omitted, every mode whose |frequency| exceeds
    --freq-threshold is scanned (this excludes the ~6 zero-frequency
    translation/rotation modes of a non-linear molecule).

--freq-threshold FLOAT   (default 10 cm-1)
--mode-amplitude FLOAT   (default 0.02 Angstrom; peak of the sine sweep)
--n-frames INT           (default 12; frames per mode, must be >= 3)
--no-mass-weight         (off by default -- see below)

MASS-WEIGHTING (on by default): ORCA's $normal_modes columns are the
eigenvectors of the *mass-weighted* Hessian, expressed in mass-weighted
coordinates q_a = sqrt(m_a) x_a. Converting a step of the normal coordinate
Q into an actual Cartesian displacement requires transforming back through
M^-1/2, i.e. dividing each atom's displacement by sqrt(m_a) (masses are read
from the .hess file's own $atoms block -- which reflects any isotope
substitution used in the frequency calculation -- falling back to standard
atomic weights only if that block is absent). Using the raw, un-mass-weighted
mode vector instead (--no-mass-weight) does not just rescale the overall
displacement -- it changes its *shape*: e.g. an H atom trans to the metal
would nominally move by the same amount as the metal itself, rather than
much further as it physically does for the same vibrational amplitude. That
distorts which nuclei end up most affected by the mode, not merely by how
much, so mass-weighting is the default and the recommended setting; the flag
is kept only for direct comparison/debugging.

------------------------------------------------------------------------------
EXAMPLE
------------------------------------------------------------------------------
python3 nja_susceptibility_shielding.py \\
    --nja-path /path/to/NJA-CFS \\
    --xyz complex.xyz \\
    --charges complex.out --charges-format orca --charge-scheme loewdin \\
    --hessian complex.hess --freq-threshold 50 --mode-amplitude 0.02 \\
    --conf f9 --metal-index 0 --temp 298 \\
    --outdir results/

(conf 'f9' = Dy3+, 'f8' = Tb3+, 'f13' = Yb3+, 'd7' = Co2+, etc. -- the number
is the number of electrons in the open d/f shell of the free ion.)
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the '3d' projection)


# ----------------------------------------------------------------------------
# Structure / plain-text charges I/O
# ----------------------------------------------------------------------------

def read_xyz(path):
    """Minimal, robust standard-XYZ reader -> (labels, coords[N,3] in Angstrom)."""
    lines = Path(path).read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"'{path}' does not look like a valid XYZ file.")
    try:
        natoms = int(lines[0].split()[0])
    except (ValueError, IndexError):
        raise ValueError(f"First line of '{path}' must be the atom count.")

    labels, coords = [], []
    for line in lines[2:2 + natoms]:
        parts = line.split()
        if len(parts) < 4:
            continue
        labels.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if len(labels) != natoms:
        raise ValueError(
            f"'{path}' declares {natoms} atoms but {len(labels)} were read."
        )
    return labels, np.array(coords, dtype=float)


def read_charges_simple(path, labels):
    """Read one charge per atom (same order as `labels`) from a plain text file."""
    charges, check_labels = [], []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) == 1:
            charges.append(float(parts[0]))
            check_labels.append(None)
        else:
            check_labels.append(parts[0])
            charges.append(float(parts[1]))

    if len(charges) != len(labels):
        raise ValueError(
            f"'{path}' has {len(charges)} charge entries but the XYZ file "
            f"has {len(labels)} atoms; they must match 1:1 (metal included)."
        )
    for i, (lab_x, lab_c) in enumerate(zip(labels, check_labels)):
        if lab_c is not None and lab_c != lab_x:
            print(f"WARNING: charges file label '{lab_c}' != xyz label "
                  f"'{lab_x}' at line {i + 1}; proceeding anyway.")
    return np.array(charges, dtype=float)


def read_charges_orca(path, labels, scheme='mulliken'):
    """
    Read atomic partial charges from an ORCA output file.

    Looks for the LAST occurrence of the requested population-analysis
    block ('MULLIKEN ATOMIC CHARGES' or 'LOEWDIN ATOMIC CHARGES') -- i.e.
    the converged/final one if the job printed several -- and matches its
    entries 1:1, in order, against `labels` (the XYZ atom order).
    """
    tag = {'mulliken': 'MULLIKEN ATOMIC CHARGES',
           'loewdin': 'LOEWDIN ATOMIC CHARGES'}.get(scheme.lower())
    if tag is None:
        raise ValueError("--charge-scheme must be 'mulliken' or 'loewdin'.")

    text = Path(path).read_text().splitlines()
    start = None
    for i, line in enumerate(text):
        if tag in line.upper():
            start = i  # keep overwriting -> ends up at the LAST occurrence
    if start is None:
        raise ValueError(f"Could not find a '{tag}' section in '{path}'.")

    charges, read_labels = [], []
    i = start + 1
    # skip the underline ('----...') and any blank lines right after the title
    while i < len(text) and (set(text[i].strip()) <= {'-'} or not text[i].strip()):
        i += 1
    while i < len(text):
        line = text[i].strip()
        if not line or line.lower().startswith('sum of atomic charges'):
            break
        # typical line: "   0 Dy :    1.234567"
        parts = line.replace(':', ' ').split()
        if len(parts) < 3:
            break
        read_labels.append(parts[1])
        charges.append(float(parts[-1]))
        i += 1

    if len(charges) != len(labels):
        raise ValueError(
            f"'{path}': found {len(charges)} {scheme} charges but the XYZ "
            f"file has {len(labels)} atoms."
        )
    for i, (lab_x, lab_c) in enumerate(zip(labels, read_labels)):
        if lab_c != lab_x:
            print(f"WARNING: ORCA charge label '{lab_c}' != xyz label "
                  f"'{lab_x}' at atom {i}; proceeding anyway (check atom order).")
    return np.array(charges, dtype=float)


def read_charges(path, labels, fmt='simple', scheme='mulliken'):
    if fmt == 'simple':
        return read_charges_simple(path, labels)
    elif fmt == 'orca':
        return read_charges_orca(path, labels, scheme=scheme)
    raise ValueError("--charges-format must be 'simple' or 'orca'.")


# ----------------------------------------------------------------------------
# ORCA .hess reader (frequencies + normal modes)
# ----------------------------------------------------------------------------

def _read_orca_block_matrix(lines, idx, square_header):
    """
    Parses one of ORCA's column-blocked matrices ($hessian or $normal_modes).
    `lines[idx]` must be the dimension header line right after the tag.
    Returns (matrix, next_idx).
    """
    header = lines[idx].split()
    idx += 1
    if square_header:
        nrows, ncols = int(header[0]), int(header[1])
    else:
        nrows = ncols = int(header[0])

    mat = np.zeros((nrows, ncols))
    col = 0
    while col < ncols:
        col_labels = lines[idx].split()
        idx += 1
        nblock = len(col_labels)
        for r in range(nrows):
            row = lines[idx].split()
            idx += 1
            vals = [float(x) for x in row[1:1 + nblock]]
            mat[r, col:col + nblock] = vals
        col += nblock
    return mat, idx


def read_orca_hessian(path):
    """
    Reads an ORCA .hess file.

    Returns a dict with:
      'hessian'   : (3N,3N) Cartesian Hessian, or None if absent
      'freqs'     : (Nmodes,) array of frequencies in cm-1, or None
      'modes'     : (3N, Nmodes) array; column k is the (raw, ORCA-normalised)
                    Cartesian displacement pattern of mode k
      'atoms'     : list of element labels from the $atoms block
      'natoms'    : number of atoms
    """
    lines = Path(path).read_text().splitlines()
    hessian = freqs = modes = None
    atoms = []
    masses = []

    i = 0
    while i < len(lines):
        tag = lines[i].strip()
        if tag == '$hessian':
            i += 1
            hessian, i = _read_orca_block_matrix(lines, i, square_header=False)
            continue
        if tag == '$vibrational_frequencies':
            i += 1
            n = int(lines[i].split()[0])
            i += 1
            freqs = np.zeros(n)
            for k in range(n):
                parts = lines[i].split()
                i += 1
                freqs[k] = float(parts[1])
            continue
        if tag == '$normal_modes':
            i += 1
            modes, i = _read_orca_block_matrix(lines, i, square_header=True)
            continue
        if tag == '$atoms':
            # format: "El  mass  x  y  z" per atom (mass is the *actual*
            # isotopic mass ORCA used for the frequency calculation, so it
            # is preferred over a generic element mass table)
            i += 1
            n = int(lines[i].split()[0])
            i += 1
            for k in range(n):
                parts = lines[i].split()
                i += 1
                atoms.append(parts[0])
                masses.append(float(parts[1]))
            continue
        i += 1

    if modes is None:
        raise ValueError(f"No '$normal_modes' section found in '{path}'.")
    natoms = modes.shape[0] // 3
    masses = np.array(masses) if masses else None
    return dict(hessian=hessian, freqs=freqs, modes=modes, atoms=atoms,
                masses=masses, natoms=natoms)


# Fallback standard atomic weights (amu), used only if a .hess file's
# $atoms block is missing or incomplete. ORCA's own $atoms masses (which
# may reflect isotope substitutions) are always preferred when present.
ELEMENT_MASSES = {
    'H': 1.00794, 'D': 2.01410, 'He': 4.002602, 'Li': 6.941, 'Be': 9.012182,
    'B': 10.811, 'C': 12.0107, 'N': 14.0067, 'O': 15.9994, 'F': 18.9984032,
    'Ne': 20.1797, 'Na': 22.98976928, 'Mg': 24.305, 'Al': 26.9815386,
    'Si': 28.0855, 'P': 30.973762, 'S': 32.065, 'Cl': 35.453, 'Ar': 39.948,
    'K': 39.0983, 'Ca': 40.078, 'Sc': 44.955912, 'Ti': 47.867, 'V': 50.9415,
    'Cr': 51.9961, 'Mn': 54.938045, 'Fe': 55.845, 'Co': 58.933195,
    'Ni': 58.6934, 'Cu': 63.546, 'Zn': 65.38, 'Ga': 69.723, 'Ge': 72.64,
    'As': 74.9216, 'Se': 78.96, 'Br': 79.904, 'Kr': 83.798, 'Rb': 85.4678,
    'Sr': 87.62, 'Y': 88.90585, 'Zr': 91.224, 'Nb': 92.90638, 'Mo': 95.96,
    'Ru': 101.07, 'Rh': 102.9055, 'Pd': 106.42, 'Ag': 107.8682, 'Cd': 112.411,
    'In': 114.818, 'Sn': 118.71, 'Sb': 121.76, 'Te': 127.6, 'I': 126.90447,
    'Xe': 131.293, 'Cs': 132.9054519, 'Ba': 137.327, 'La': 138.90547,
    'Ce': 140.116, 'Pr': 140.90765, 'Nd': 144.242, 'Pm': 145.0, 'Sm': 150.36,
    'Eu': 151.964, 'Gd': 157.25, 'Tb': 158.92535, 'Dy': 162.5, 'Ho': 164.93032,
    'Er': 167.259, 'Tm': 168.93421, 'Yb': 173.054, 'Lu': 174.9668, 'Hf': 178.49,
    'Ta': 180.94788, 'W': 183.84, 'Re': 186.207, 'Os': 190.23, 'Ir': 192.217,
    'Pt': 195.084, 'Au': 196.966569, 'Hg': 200.59, 'Tl': 204.3833, 'Pb': 207.2,
    'Bi': 208.9804, 'U': 238.02891,
}


def fallback_masses(labels):
    masses = []
    for lab in labels:
        m = ELEMENT_MASSES.get(lab)
        if m is None:
            print(f"WARNING: no built-in mass for element '{lab}'; using 1.0 amu.")
            m = 1.0
        masses.append(m)
    return np.array(masses)


def mode_displacement_vectors(hess_data, mode_indices, mass_weight=True, xyz_labels=None):
    """
    For each requested mode index, returns a (natoms,3) unit-norm Cartesian
    displacement pattern to be used as dx = amplitude * pattern.

    mass_weight=True (recommended, and the default):
        ORCA's $normal_modes columns are the eigenvectors L of the
        mass-weighted Hessian (M^-1/2 H M^-1/2) L = L * lambda, i.e. they
        live in mass-weighted coordinates q_a = sqrt(m_a) * x_a. To turn a
        unit step of the associated normal coordinate Q into an actual
        Cartesian displacement one must transform back with M^-1/2:

            dx_a = L_a / sqrt(m_a)   (per atom a, 3 Cartesian components)

        Skipping this step (i.e. moving every atom by the same raw
        Cartesian amount, regardless of its mass) is *not* how the atoms
        actually move along that normal mode: light atoms (e.g. H) swing
        much further than heavy ones (e.g. the lanthanide) for the same
        vibrational quantum/energy, and the mode vector as stored already
        encodes that -- dividing by sqrt(mass) is what recovers it. Not
        mass-weighting silently distorts the *shape* of the displacement
        (and hence which nuclei "feel" most of the distortion), not just
        its overall scale.

        After the M^-1/2 transformation the resulting Cartesian pattern is
        renormalized to unit Euclidean norm, so --mode-amplitude keeps the
        simple meaning "typical Cartesian displacement magnitude of the
        pattern, in Angstrom"; the *relative* motion of each atom within
        that pattern is still the physically mass-weighted one.

    mass_weight=False:
        Use the raw $normal_modes columns directly (renormalized to unit
        Cartesian norm), i.e. every atom nominally displaced by the same
        order of magnitude irrespective of its mass. Kept only for
        comparison/debugging.
    """
    modes = hess_data['modes']
    natoms = hess_data['natoms']

    inv_sqrt_m = None
    if mass_weight:
        masses = hess_data.get('masses')
        if masses is None or len(masses) != natoms:
            labels_for_mass = hess_data.get('atoms') or xyz_labels
            if not labels_for_mass or len(labels_for_mass) != natoms:
                raise ValueError("Cannot mass-weight: no atomic masses found in the "
                                  ".hess file and no matching XYZ labels available.")
            print("NOTE: no per-atom masses in the .hess $atoms block; falling back "
                  "to standard atomic weights from element labels.")
            masses = fallback_masses(labels_for_mass)
        inv_sqrt_m = 1.0 / np.sqrt(masses)

    out = {}
    for k in mode_indices:
        if k < 0 or k >= modes.shape[1]:
            raise ValueError(f"Mode index {k} out of range (0..{modes.shape[1] - 1}).")
        vec = modes[:, k].reshape(natoms, 3).copy()
        if mass_weight:
            vec *= inv_sqrt_m[:, None]
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            raise ValueError(f"Mode {k} has (numerically) zero norm; cannot use it.")
        out[k] = vec / norm
    return out


def select_mode_indices(hess_data, requested, freq_threshold):
    if requested:
        return [int(x) for x in requested.split(',') if x.strip() != '']
    freqs = hess_data['freqs']
    nmodes = hess_data['modes'].shape[1]
    if freqs is None:
        print("WARNING: no frequency information in the .hess file; "
              "scanning ALL modes (including translations/rotations).")
        return list(range(nmodes))
    return [k for k in range(nmodes) if abs(freqs[k]) >= freq_threshold]


# ----------------------------------------------------------------------------
# Electronic-structure setup (NJA-CFS)
# ----------------------------------------------------------------------------

def free_ion_parameters(nja, conf):
    """Pick sensible default Slater-Condon (F2,F4,F6) and SOC (zeta) parameters."""
    if conf[0] == 'f':
        try:
            par = nja.free_ion_param_f(conf)
        except KeyError:
            par = nja.free_ion_param_f_HF(conf)
        return dict(F2=par['F2'], F4=par['F4'], F6=par.get('F6', 0), zeta=par['zeta'])
    elif conf[0] == 'd':
        par = nja.free_ion_param_AB(conf)
        return dict(F2=par.get('F2', 0), F4=par.get('F4', 0), F6=0, zeta=par['zeta'])
    else:
        raise ValueError(f"Unrecognised configuration '{conf}' (expected d1-d9 or f1-f13).")


def build_ligand_data(labels, coords, charges, metal_index):
    """
    Builds the [label, x, y, z, charge] array (ligand coordinates relative
    to the metal ion) expected by nja.calc_Bkq, plus convenience outputs.
    """
    metal_xyz = coords[metal_index]
    mask = np.array([i != metal_index for i in range(len(labels))])
    lig_labels = [l for l, m in zip(labels, mask) if m]
    lig_coords = coords[mask] - metal_xyz
    lig_charges = charges[mask]

    data = np.zeros((lig_coords.shape[0], 5), dtype=object)
    data[:, 0] = lig_labels
    data[:, 1:4] = lig_coords
    data[:, 4] = -lig_charges
    return data, lig_labels, lig_coords, mask


def build_calculation(nja, conf, data, ground_only, F2, F4, F6, zeta):
    """Build+diagonalise the CF Hamiltonian, returns (calc, basis, LF_matrix, projected)."""
    dic_bkq = nja.calc_Bkq(data, conf)
    dic = dict(F2=F2, F4=F4, F6=F6, zeta=zeta, dic_bkq=dic_bkq)

    calc = nja.calculation(conf, ground_only=ground_only, TAB=True, wordy=False)
    _, projected = calc.MatrixH(
        ['Hee', 'Hso', 'Hcf'], **dic,
        eig_opt=False, wordy=False,
        ground_proj=True, return_proj=True,
        save_label=True, save_LF=True,
    )

    # NJA-CFS writes the zero-field ligand-field matrix to ./matrix_LF.npy
    lf_path = Path('matrix_LF.npy')
    LF_matrix = np.load(lf_path, allow_pickle=True).astype(np.complex128)
    lf_path.unlink(missing_ok=True)

    basis = calc.basis.astype(np.float64)
    return calc, basis, LF_matrix, projected


def susceptibility_tensor(nja, basis, LF_matrix, temp, delta=1.0):
    """
    Full 3x3 magnetic susceptibility tensor (SI, m^3, single-molecule volume
    susceptibility) via numerical differentiation of the magnetic moment
    w.r.t. the applied field (Ridders' method), as implemented in NJA-CFS.
    """
    field = np.array([[0.0, 0.0, 0.0]])
    chi_tensor, err_tensor = nja.susceptibility_B_ord1(field, temp, basis, LF_matrix, delta=delta)
    chi_tensor = 0.5 * (chi_tensor + chi_tensor.T)  # symmetrize (numerical noise)
    return chi_tensor, err_tensor


# ----------------------------------------------------------------------------
# Nuclear dipolar (pseudocontact) shielding tensor
# ----------------------------------------------------------------------------

ANGSTROM = 1e-10  # m


def dipolar_shielding_tensor(chi_tensor, r_vec_angstrom):
    """
    Point-dipole (Kurland-McGarvey) dipolar/pseudocontact nuclear shielding
    tensor produced at a nucleus by the metal's magnetic susceptibility
    tensor `chi_tensor` (SI, m^3), given the metal->nucleus vector
    `r_vec_angstrom` (Angstrom):

        sigma = - 1/(4*pi*r^5) * (3 r r^T - r^2 * I) . chi        (dimensionless)

    Sign convention: sigma is a *shielding* (B_local = (1-sigma) B0); the
    corresponding pseudocontact *shift* is delta_pc = -Tr(sigma)/3 (in ppm
    once multiplied by 1e6).
    """
    r = np.asarray(r_vec_angstrom, dtype=float) * ANGSTROM
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-3 * ANGSTROM:
        raise ValueError("Nucleus coincides with the metal center; undefined dipolar term.")
    dyad = 3.0 * np.outer(r, r) - (r_norm ** 2) * np.eye(3)
    return -(dyad @ chi_tensor) / (4.0 * np.pi * r_norm ** 5)


def principal_components(tensor_3x3):
    """Return (eigenvalues sorted ascending, eigenvectors as columns)."""
    sym = 0.5 * (tensor_3x3 + tensor_3x3.T)
    return np.linalg.eigh(sym)


# ----------------------------------------------------------------------------
# One-shot evaluation: chi tensor + shielding tensors for a given geometry
# ----------------------------------------------------------------------------

def evaluate_structure(nja, labels, coords, charges, metal_index, conf,
                        F2, F4, F6, zeta, ground_only, temp, target_mask):
    """
    Runs the full NJA-CFS pipeline for one geometry and returns:
      chi_tensor              : (3,3) SI, m^3
      sigma_tensors           : list of (3,3) shielding tensors, one per
                                 target nucleus (in the order given by
                                 target_mask over the ligand list)
      lig_labels, lig_coords  : bookkeeping (ligand-frame, relative to metal)
      lig_charges              : point charges used, same order as lig_labels
      projected, ground_only   : the ground-state |J,MJ> composition returned
                                 by NJA-CFS, and whether `calc` was built with
                                 ground_only=True (needed to unpack `projected`)
    """
    data, lig_labels, lig_coords, mask = build_ligand_data(labels, coords, charges, metal_index)
    lig_charges = charges[mask]
    _, basis, LF_matrix, projected = build_calculation(nja, conf, data, ground_only, F2, F4, F6, zeta)
    chi_tensor, _ = susceptibility_tensor(nja, basis, LF_matrix, temp)

    sigma_tensors = []
    for lab, r_vec, take in zip(lig_labels, lig_coords, target_mask):
        if take:
            sigma_tensors.append(dipolar_shielding_tensor(chi_tensor, r_vec))
    return chi_tensor, sigma_tensors, lig_labels, lig_coords, lig_charges, projected


# ----------------------------------------------------------------------------
# Ground-state electron-density shape (Sievers-like) + point-charge plot
# ----------------------------------------------------------------------------

# Second-order ("k=2") shape moment at the stretched (MJ = +/-J) state of
# each trivalent lanthanide's Hund's-rule ground term, from
# A. J. Sievers, "Asphericity of 4f-Shells in their Hund's Rule Ground
# States", Table 1 (values also present, unused, in NJA-CFS's own
# `A_table()`). Indexed by number of 4f electrons (i.e. the integer in
# 'f1'..'f13'). f6 (Eu3+, ground term 7F0, J=0) has no orientational
# anisotropy and is omitted (A2 = 0); f7 (Gd3+, 8S7/2) is an S-state ion and
# is correctly isotropic (A2 = 0) in the table.
SIEVERS_A2_STRETCHED = {
    1: (2.5, -0.2857),   # Ce3+  2F5/2
    2: (4.0, -0.2941),   # Pr3+  3H4
    3: (4.5, -0.1157),   # Nd3+  4I9/2
    4: (4.0, 0.1080),    # Pm3+  5I4
    5: (2.5, 0.2063),    # Sm3+  6H5/2
    6: (0.0, 0.0),       # Eu3+  7F0  (J=0)
    7: (3.5, 0.0),       # Gd3+  8S7/2 (isotropic)
    8: (6.0, -0.3333),   # Tb3+  7F6
    9: (7.5, -0.3333),   # Dy3+  6H15/2
    10: (8.0, -0.1333),  # Ho3+  5I8
    11: (7.5, 0.1333),   # Er3+  4I15/2
    12: (6.0, 0.3333),   # Tm3+  3H6
    13: (3.5, 0.3333),   # Yb3+  2F7/2
}


def _frac_to_float(s):
    s = s.strip()
    if '/' in s:
        num, den = s.split('/')
        return float(num) / float(den)
    return float(s)


def _parse_JM_label(label):
    """
    Parses an NJA-CFS J-resolved state label, e.g. '6H(15/2) 7.5' or
    '7F(6) -6', into (J, MJ) floats.
    """
    m = re.search(r'\(([^)]+)\)\s*(-?[\d./]+)\s*$', label)
    if not m:
        raise ValueError(f"Could not parse J/MJ from state label '{label}'.")
    return _frac_to_float(m.group(1)), _frac_to_float(m.group(2))


def sievers_A2(nel, J, MJ):
    """
    Sievers second-order shape moment for an arbitrary |J,MJ> state, scaled
    from the ion's tabulated stretched-state (MJ=+/-J_ref) value using the
    standard M-dependence of a rank-2 tensor operator within a J multiplet:

        A2(J,MJ) = A2_ref * [3*MJ^2 - J(J+1)] / [J_ref*(2*J_ref-1)]

    This is exact when J == J_ref (i.e. for any state that stays within the
    ion's ground J multiplet, which is always true with --ground-only, and
    in practice true to a very good approximation otherwise, since spin-
    orbit coupling keeps J well defined for the lanthanides). If a state
    happens to belong to a different J (only possible with intermediate
    J-mixing, i.e. without --ground-only), the same ion-specific A2_ref is
    still used as a reasonable approximation, since Sievers' table only
    tabulates each ion's ground-J term.
    """
    if nel not in SIEVERS_A2_STRETCHED:
        return 0.0
    J_ref, A2_ref = SIEVERS_A2_STRETCHED[nel]
    denom = J_ref * (2.0 * J_ref - 1.0)
    if denom == 0:
        return 0.0
    return A2_ref * (3.0 * MJ ** 2 - J * (J + 1.0)) / denom


def ground_state_A2_mixed(projected, calc_is_ground_only, nel):
    """
    Population-weighted (incoherent mixture) second-order Sievers shape
    moment of the lowest-energy (ground) crystal-field state, from its
    |J,MJ> decomposition as returned by NJA-CFS
    (calc.MatrixH(..., ground_proj=True, return_proj=True)).

    Returns (A2_mixed, components), where components is a list of
    (label, J, MJ, population_fraction, A2_component) for transparency/output.
    """
    proj_ground = projected[1] if calc_is_ground_only else projected[1][1]
    A2_mixed = 0.0
    components = []
    for label, pct in proj_ground.items():
        J, MJ = _parse_JM_label(label)
        a2 = sievers_A2(nel, J, MJ)
        w = pct / 100.0
        A2_mixed += w * a2
        components.append((label, J, MJ, w, a2))
    components.sort(key=lambda c: c[3], reverse=True)
    return A2_mixed, components


def plot_charge_and_density(outdir, metal_label, lig_labels, lig_coords, lig_charges,
                             A2_mixed, density_scale=1.0, n_theta=60, n_phi=60,
                             elev=20, azim=-60, file_name='charge_density_sievers.png'):
    """
    3D visualization of the point charges used in the crystal-field model
    together with the shape of the ground crystal-field state's 4f electron
    density in the Sievers-like, population-mixed approximation:

        rho(theta) = 1 + A2_mixed * (3*cos(theta)^2 - 1)     (axially symmetric)

    Both are drawn centered on the metal ion, in the same lab frame used to
    build the crystal-field Hamiltonian (z = the XYZ file's z axis, which is
    the CF quantization axis since calc_Bkq works directly in that frame).
    `density_scale` (Angstrom) sets the size of the (dimensionless) rho(theta)
    surface so it is visually comparable to the ligand distances; it is a
    plotting choice, not a physical radius.
    """
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta_g, phi_g = np.meshgrid(theta, phi)
    rho = 1.0 + A2_mixed * (3.0 * np.cos(theta_g) ** 2 - 1.0)
    rho = np.clip(rho, 0.0, None)  # radii must stay non-negative
    X = density_scale * rho * np.sin(theta_g) * np.cos(phi_g)
    Y = density_scale * rho * np.sin(theta_g) * np.sin(phi_g)
    Z = density_scale * rho * np.cos(theta_g)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, color='goldenrod', alpha=0.45, linewidth=0,
                     antialiased=True, shade=True, zorder=1)

    coords = np.asarray(lig_coords, dtype=float)
    charges = np.asarray(lig_charges, dtype=float)
    vmax = max(np.abs(charges).max(), 1e-6) if len(charges) else 1.0
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                     c=charges, cmap='coolwarm', vmin=-vmax, vmax=vmax,
                     s=120, edgecolors='k', linewidths=0.5, depthshade=True, zorder=5)
    for lab, (x, y, z), q in zip(lig_labels, coords, charges):
        ax.text(x, y, z, f'{lab} ({q:+.2f})', fontsize=7)

    ax.scatter([0], [0], [0], c='black', s=180, marker='*', zorder=6)
    ax.text(0, 0, 0, f'  {metal_label}', fontsize=9, weight='bold')

    r_ligs = [np.linalg.norm(c) for c in coords] if len(coords) else [1.0]
    lim = 1.15 * max(max(r_ligs), density_scale * (1.0 + abs(A2_mixed) * 2.0))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('x (\u00c5)')
    ax.set_ylabel('y (\u00c5)')
    ax.set_zlabel('z (\u00c5)')

    shape_word = 'oblate' if A2_mixed < 0 else ('prolate' if A2_mixed > 0 else 'isotropic')
    ax.set_title(f"Ground-state 4f density (Sievers-like, {shape_word},\n"
                 f"population-mixed A2 = {A2_mixed:+.4f}) and point charges", fontsize=10)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cb.set_label('point charge (e)')
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()

    out_path = Path(outdir) / file_name
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Compute the magnetic susceptibility tensor and nuclear "
                    "dipolar shielding tensors with NJA-CFS, optionally scanning "
                    "vibrational modes from an ORCA Hessian.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--nja-path', default='.', help="Path to the NJA-CFS repository "
                    "(checked out at the 'sph_harm_y' branch); default: current dir.")
    p.add_argument('--xyz', required=True, help="XYZ file with the full structure "
                    "(metal + ligands + NMR-active nuclei).")
    p.add_argument('--charges', required=True,
                    help="Charges source: plain text file (--charges-format simple, "
                         "default) or an ORCA output file (--charges-format orca).")
    p.add_argument('--charges-format', choices=['simple', 'orca'], default='simple')
    p.add_argument('--charge-scheme', choices=['mulliken', 'loewdin'], default='mulliken',
                    help="Population analysis to use when --charges-format orca.")
    p.add_argument('--metal-index', type=int, default=0,
                    help="0-based index of the paramagnetic metal ion in the XYZ file. Default 0.")
    p.add_argument('--conf', required=True,
                    help="Open-shell configuration of the metal ion, e.g. 'f9' for Dy3+, "
                         "'f8' for Tb3+, 'd7' for Co2+.")
    p.add_argument('--temp', type=float, default=298.0, help="Temperature in K. Default 298.")
    p.add_argument('--ground-only', action='store_true',
                    help="Restrict the basis to the ground LS/J multiplet (faster, "
                         "recommended for lanthanides).")
    p.add_argument('--F2', type=float, default=None, help="Override Slater-Condon F2 (cm-1).")
    p.add_argument('--F4', type=float, default=None, help="Override Slater-Condon F4 (cm-1).")
    p.add_argument('--F6', type=float, default=None, help="Override Slater-Condon F6 (cm-1).")
    p.add_argument('--zeta', type=float, default=None, help="Override spin-orbit coupling (cm-1).")
    p.add_argument('--nuclei', default=None,
                    help="Comma separated list of element labels for which the shielding "
                         "tensor is computed (default: all non-metal atoms).")
    # vibrational mode scan
    p.add_argument('--hessian', default=None,
                    help="ORCA .hess file; if given, scans the selected vibrational modes.")
    p.add_argument('--modes', default=None,
                    help="Comma separated 0-based mode indices to scan (default: all "
                         "modes above --freq-threshold).")
    p.add_argument('--freq-threshold', type=float, default=10.0,
                    help="cm-1; modes with |freq| below this are treated as "
                         "translations/rotations and skipped by default. Default 10.")
    p.add_argument('--mode-amplitude', type=float, default=0.02,
                    help="Peak Cartesian displacement (Angstrom) of the sinusoidal "
                         "sweep along each mode (see --n-frames). Default 0.02.")
    p.add_argument('--n-frames', type=int, default=12,
                    help="Number of frames used to sample each mode: frame i "
                         "(i=0..n-1) is displaced by "
                         "amplitude*sin(2*pi*i/(n-1)) along the mode, i.e. one full "
                         "period starting and ending at the equilibrium geometry. "
                         "Default 12.")
    p.add_argument('--no-mass-weight', action='store_true',
                    help="Use the raw Cartesian $normal_modes vectors (renormalized "
                         "to unit norm) instead of mass-weighting them back from the "
                         "mass-weighted-Hessian eigenvectors. Not recommended; see "
                         "mode_displacement_vectors() docstring.")
    # ground-state electron-density (Sievers-like) + charge plot
    p.add_argument('--plot-density', action='store_true',
                    help="Plot the ground crystal-field state's Sievers-like, "
                         "population-mixed 4f electron-density shape together with "
                         "the point charges (f-element configurations only).")
    p.add_argument('--density-scale', type=float, default=1.0,
                    help="Angstrom; visual size of the (dimensionless) density "
                         "surface in the plot. Default 1.0.")
    p.add_argument('--outdir', default='nja_output', help="Output directory.")
    args = p.parse_args()

    sys.path.insert(0, str(Path(args.nja_path).resolve()))
    try:
        import nja_cfs_red as nja
    except ImportError as e:
        sys.exit(f"Could not import nja_cfs_red from '{args.nja_path}': {e}\n"
                  f"Pass --nja-path pointing to the NJA-CFS checkout "
                  f"(branch 'sph_harm_y').")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -- read structure and charges -----------------------------------------
    labels, coords = read_xyz(args.xyz)
    charges = read_charges(args.charges, labels, fmt=args.charges_format,
                            scheme=args.charge_scheme)

    if not (0 <= args.metal_index < len(labels)):
        sys.exit(f"--metal-index {args.metal_index} out of range (0..{len(labels) - 1}).")

    metal_label = labels[args.metal_index]
    print(f"Metal center: {metal_label} (index {args.metal_index}) at {coords[args.metal_index]} A")
    print(f"Charges source: {args.charges} ({args.charges_format}"
          + (f", {args.charge_scheme}" if args.charges_format == 'orca' else "") + ")")

    # -- free-ion / Slater-Condon parameters ---------------------------------
    default_pars = free_ion_parameters(nja, args.conf)
    F2 = args.F2 if args.F2 is not None else default_pars['F2']
    F4 = args.F4 if args.F4 is not None else default_pars['F4']
    F6 = args.F6 if args.F6 is not None else default_pars['F6']
    zeta = args.zeta if args.zeta is not None else default_pars['zeta']
    print(f"Configuration {args.conf}: F2={F2}, F4={F4}, F6={F6}, zeta={zeta} (cm-1)")

    # target-nucleus mask, built against the ligand-ordered label list
    _, lig_labels_ref, _, _ = build_ligand_data(labels, coords, charges, args.metal_index)
    if args.nuclei:
        wanted = set(s.strip() for s in args.nuclei.split(','))
        target_mask = np.array([lab in wanted for lab in lig_labels_ref])
    else:
        target_mask = np.ones(len(lig_labels_ref), dtype=bool)
    target_idx = [i for i, t in enumerate(target_mask) if t]

    # ==========================================================================
    # Equilibrium geometry: chi tensor + shielding tensors
    # ==========================================================================
    chi_eq, sigma_eq, lig_labels, lig_coords, lig_charges, projected_eq = evaluate_structure(
        nja, labels, coords, charges, args.metal_index, args.conf,
        F2, F4, F6, zeta, args.ground_only, args.temp, target_mask)

    chi_eigval, _ = principal_components(chi_eq)
    chi_iso = np.trace(chi_eq) / 3.0

    print("\n=== Magnetic susceptibility tensor (SI, m^3, single molecule) ===")
    print(chi_eq)
    print(f"isotropic chi = {chi_iso:.6e} m^3  "
          f"({chi_iso * 6.02214076e23 * 1e6:.6e} cm^3/mol)")
    print(f"principal values (m^3): {chi_eigval}")

    np.savetxt(outdir / 'chi_tensor.txt', chi_eq,
               header='Magnetic susceptibility tensor, SI units (m^3), single molecule')

    print("\n=== Nuclear dipolar (pseudocontact) shielding tensors (equilibrium) ===")
    shift_eq = {}
    with open(outdir / 'shielding_tensors.txt', 'w') as f:
        f.write("# idx label  r(A)    delta_pc(ppm)   sigma_xx sigma_xy sigma_xz "
                "sigma_yx sigma_yy sigma_yz sigma_zx sigma_zy sigma_zz  (ppm)\n")
        for j, i in enumerate(target_idx):
            lab, r_vec, sigma = lig_labels[i], lig_coords[i], sigma_eq[j]
            r_norm = np.linalg.norm(r_vec)
            iso_ppm = -np.trace(sigma) / 3.0 * 1e6
            shift_eq[i] = iso_ppm
            s_eigval, _ = principal_components(sigma)
            print(f"\nNucleus {i:3d} {lab:>3s}  r = {r_norm:7.3f} A")
            print(sigma * 1e6, "  [ppm]")
            print(f"  principal values (ppm): {s_eigval * 1e6}")
            print(f"  pseudocontact shift delta_pc = {iso_ppm:.4f} ppm")
            sigma_ppm = (sigma * 1e6).flatten()
            f.write(f"{i:4d} {lab:>4s} {r_norm:8.4f} {iso_ppm:12.4f} "
                    + " ".join(f"{v:12.5f}" for v in sigma_ppm) + "\n")

    print(f"\nEquilibrium results written to: {outdir}/")
    print("  - chi_tensor.txt, shielding_tensors.txt")

    # ==========================================================================
    # Optional: ground-state Sievers-like density + charge plot
    # ==========================================================================
    if args.plot_density:
        if args.conf[0] != 'f':
            print("\nWARNING: --plot-density only implemented for f-element "
                  "configurations (Sievers table is f-electron specific); skipping.")
        else:
            nel = int(args.conf[1:])
            A2_mixed, components = ground_state_A2_mixed(
                projected_eq, args.ground_only, nel)
            print(f"\n=== Ground-state Sievers-like density (population-mixed) ===")
            print(f"{'label':>14} {'J':>6} {'MJ':>6} {'pop.(%)':>9} {'A2_comp':>10}")
            for label, J, MJ, w, a2 in components:
                print(f"{label:>14} {J:6.1f} {MJ:6.1f} {w * 100:9.3f} {a2:10.4f}")
            shape_word = ('oblate' if A2_mixed < 0 else
                          'prolate' if A2_mixed > 0 else 'isotropic')
            print(f"Population-mixed A2 = {A2_mixed:+.4f}  -> {shape_word} "
                  f"ground-state 4f density")

            plot_path = plot_charge_and_density(
                outdir, metal_label, lig_labels, lig_coords, lig_charges,
                A2_mixed, density_scale=args.density_scale)

            comp_path = outdir / 'ground_state_density_composition.txt'
            with open(comp_path, 'w') as f:
                f.write("# label J MJ population(%) A2_component\n")
                for label, J, MJ, w, a2 in components:
                    f.write(f"{label} {J:.2f} {MJ:.2f} {w * 100:.4f} {a2:.6f}\n")
                f.write(f"# population-mixed A2 = {A2_mixed:.6f} ({shape_word})\n")

            print(f"\nDensity/charge plot written to: {plot_path}")
            print(f"Composition table written to: {comp_path}")

    # ==========================================================================
    # Optional: vibrational-mode scan
    # ==========================================================================
    if args.hessian is None:
        return

    hess_data = read_orca_hessian(args.hessian)
    if hess_data['natoms'] != len(labels):
        print(f"WARNING: .hess file describes {hess_data['natoms']} atoms but the "
              f"XYZ file has {len(labels)}; make sure they refer to the same "
              f"structure/atom order.")

    mode_indices = select_mode_indices(hess_data, args.modes, args.freq_threshold)
    if not mode_indices:
        print("No vibrational modes selected (check --modes / --freq-threshold); "
              "skipping mode scan.")
        return

    mass_weight = not args.no_mass_weight
    disp_vectors = mode_displacement_vectors(
        hess_data, mode_indices, mass_weight=mass_weight, xyz_labels=labels)
    amp = args.mode_amplitude
    n_frames = args.n_frames
    if n_frames < 3:
        sys.exit("--n-frames must be >= 3.")
    freqs = hess_data['freqs']

    # frame i (i=0..n-1) sits at Q_i = amplitude*sin(2*pi*i/(n-1)) along the
    # mode: this sweeps a full period, starting and ending at the
    # equilibrium (undisplaced) geometry.
    i_arr = np.arange(n_frames)
    Q_frac = np.sin(2.0 * np.pi * i_arr / (n_frames - 1))  # dimensionless, in [-1,1]
    Q = Q_frac * amp                                       # Angstrom
    denom = np.dot(Q, Q)  # for the through-origin linear-regression slope below

    print(f"\n=== Vibrational-mode scan: {len(mode_indices)} mode(s), "
          f"{n_frames} frames/mode, peak amplitude = {amp} A, "
          f"mass-weighted = {mass_weight} ===")

    mode_rows = []
    shift_rows = {i: [] for i in target_idx}  # nucleus -> list of (mode, freq, slope)

    for k in mode_indices:
        freq = freqs[k] if freqs is not None else float('nan')
        vec = disp_vectors[k]

        chi_traj = np.zeros((n_frames, 3, 3))
        shift_traj = {i: np.zeros(n_frames) for i in target_idx}

        for f_idx in range(n_frames):
            coords_f = coords + Q[f_idx] * vec
            chi_f, sigma_f, _, _, _, _ = evaluate_structure(
                nja, labels, coords_f, charges, args.metal_index, args.conf,
                F2, F4, F6, zeta, args.ground_only, args.temp, target_mask)
            chi_traj[f_idx] = chi_f
            for j, i in enumerate(target_idx):
                shift_traj[i][f_idx] = -np.trace(sigma_f[j]) / 3.0 * 1e6  # ppm

        # linear-response coupling constant: least-squares slope through the
        # origin of (quantity) vs Q, using every frame of the sinusoidal
        # sweep -- more robust to numerical noise than a plain 2-point
        # central difference, and it also lets you inspect the full
        # chi(Q)/shift(Q) trajectory for non-linear (anharmonic) response.
        dchi_dQ = np.tensordot(Q, chi_traj, axes=(0, 0)) / denom  # m^3 / A
        chi_iso_slope = np.trace(dchi_dQ) / 3.0
        chi_iso_traj = np.trace(chi_traj, axis1=1, axis2=2) / 3.0

        print(f"\n-- mode {k:3d}  freq = {freq:9.2f} cm-1 --")
        print("d(chi)/dQ (m^3/A, linear-regression slope over the sweep):")
        print(dchi_dQ)
        print(f"d(chi_iso)/dQ = {chi_iso_slope:.6e} m^3/A")

        mode_rows.append((k, freq, dchi_dQ.copy(), chi_iso_slope))

        # per-mode trajectory file (frame-by-frame chi_iso and shifts)
        traj_path = outdir / f'mode_{k:03d}_trajectory.txt'
        with open(traj_path, 'w') as f:
            header = "# frame  Q(A)  chi_iso(m^3)  " + \
                     " ".join(f"shift_{i}_{lig_labels[i]}(ppm)" for i in target_idx)
            f.write(header + "\n")
            for f_idx in range(n_frames):
                row = [f"{f_idx:5d}", f"{Q[f_idx]: .6f}", f"{chi_iso_traj[f_idx]: .6e}"]
                row += [f"{shift_traj[i][f_idx]: .4f}" for i in target_idx]
                f.write(" ".join(row) + "\n")

        for j, i in enumerate(target_idx):
            dshift_dQ = np.dot(Q, shift_traj[i]) / denom  # ppm / A
            shift_rows[i].append((k, freq, dshift_dQ))
            print(f"   nucleus {i:3d} {lig_labels[i]:>3s}: "
                  f"d(delta_pc)/dQ = {dshift_dQ:10.4f} ppm/A")

    # -- write mode-scan summary (slopes) --------------------------------------
    with open(outdir / 'chi_mode_derivatives.txt', 'w') as f:
        f.write("# mode  freq(cm-1)  d(chi_iso)/dQ(m^3/A)  "
                "dchi_xx dchi_xy dchi_xz dchi_yx dchi_yy dchi_yz dchi_zx dchi_zy dchi_zz (m^3/A)\n")
        for k, freq, dchi_dQ, chi_iso_slope in mode_rows:
            f.write(f"{k:4d} {freq:12.3f} {chi_iso_slope:14.6e} "
                    + " ".join(f"{v:14.6e}" for v in dchi_dQ.flatten()) + "\n")

    with open(outdir / 'shift_mode_derivatives.txt', 'w') as f:
        f.write("# nucleus_idx label mode freq(cm-1)  d(delta_pc)/dQ(ppm/A)\n")
        for i in target_idx:
            for k, freq, dshift_dQ in shift_rows[i]:
                f.write(f"{i:4d} {lig_labels[i]:>4s} {k:4d} {freq:12.3f} {dshift_dQ:14.6f}\n")

    print(f"\nMode-scan results written to: {outdir}/")
    print("  - chi_mode_derivatives.txt, shift_mode_derivatives.txt (fitted slopes)")
    print("  - mode_<k>_trajectory.txt (one per mode: full chi_iso(Q)/shift(Q) sweep)")


if __name__ == '__main__':
    main()

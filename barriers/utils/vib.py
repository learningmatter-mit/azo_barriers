from rdkit import Chem
import numpy as np
import os
import shutil

from ase import Atoms
from ase.vibrations import Vibrations
from ase.units import kg, kB, mol, J, m
from ase.thermochemistry import IdealGasThermo

from nff.utils.constants import EV_TO_AU, BOHR_RADIUS
from nff.nn.tensorgrad import hess_from_atoms as analytical_hess

PT = Chem.GetPeriodicTable()

HA2J = 4.359744E-18
BOHRS2ANG = 0.529177
SPEEDOFLIGHT = 2.99792458E8
AMU2KG = 1.660538782E-27

TEMP = 298.15
PRESSURE = 101325
IMAG_CUTOFF = -100  # cm^-1
ROTOR_CUTOFF = 50  # cm^-1
CM_TO_EV = 1.2398e-4
GAS_CONST = 8.3144621 * J / mol
B_AV = 1e-44 * kg * m ** 2


def moi_tensor(massvec, expmassvec, xyz):
    # Center of Mass
    com = np.sum(expmassvec.reshape(-1, 3) *
                 xyz.reshape(-1, 3), axis=0
                 ) / np.sum(massvec)

    # xyz shifted to COM
    xyz_com = xyz.reshape(-1, 3) - com

    # Compute elements need to calculate MOI tensor
    mass_xyz_com_sq_sum = np.sum(
        expmassvec.reshape(-1, 3) * xyz_com ** 2, axis=0)

    mass_xy = np.sum(massvec * xyz_com[:, 0] * xyz_com[:, 1], axis=0)
    mass_yz = np.sum(massvec * xyz_com[:, 1] * xyz_com[:, 2], axis=0)
    mass_xz = np.sum(massvec * xyz_com[:, 0] * xyz_com[:, 2], axis=0)

    # MOI tensor
    moi = np.array([[mass_xyz_com_sq_sum[1] + mass_xyz_com_sq_sum[2], -1 *
                     mass_xy, -1 * mass_xz],
                    [-1 * mass_xy, mass_xyz_com_sq_sum[0] +
                        mass_xyz_com_sq_sum[2], -1 * mass_yz],
                    [-1 * mass_xz, -1 * mass_yz, mass_xyz_com_sq_sum[0] +
                     mass_xyz_com_sq_sum[1]]])

    # MOI eigenvectors and eigenvalues
    moi_eigval, moi_eigvec = np.linalg.eig(moi)

    return xyz_com, moi_eigvec


def trans_rot_vec(massvec, xyz_com, moi_eigvec):

    # Mass-weighted translational vectors
    zero_vec = np.zeros([len(massvec)])
    sqrtmassvec = np.sqrt(massvec)
    expsqrtmassvec = np.repeat(sqrtmassvec, 3)

    d1 = np.transpose(np.stack((sqrtmassvec, zero_vec, zero_vec))).reshape(-1)
    d2 = np.transpose(np.stack((zero_vec, sqrtmassvec, zero_vec))).reshape(-1)
    d3 = np.transpose(np.stack((zero_vec, zero_vec, sqrtmassvec))).reshape(-1)

    # Mass-weighted rotational vectors
    big_p = np.matmul(xyz_com, moi_eigvec)

    d4 = (np.repeat(big_p[:, 1], 3).reshape(-1) *
          np.tile(moi_eigvec[:, 2], len(massvec)).reshape(-1) -
          np.repeat(big_p[:, 2], 3).reshape(-1) *
          np.tile(moi_eigvec[:, 1], len(massvec)).reshape(-1)
          ) * expsqrtmassvec

    d5 = (np.repeat(big_p[:, 2], 3).reshape(-1) *
          np.tile(moi_eigvec[:, 0], len(massvec)).reshape(-1) -
          np.repeat(big_p[:, 0], 3).reshape(-1) *
          np.tile(moi_eigvec[:, 2], len(massvec)).reshape(-1)
          ) * expsqrtmassvec

    d6 = (np.repeat(big_p[:, 0], 3).reshape(-1) *
          np.tile(moi_eigvec[:, 1], len(massvec)).reshape(-1) -
          np.repeat(big_p[:, 1], 3).reshape(-1) *
          np.tile(moi_eigvec[:, 0], len(massvec)).reshape(-1)
          ) * expsqrtmassvec

    d1_norm = d1 / np.linalg.norm(d1)
    d2_norm = d2 / np.linalg.norm(d2)
    d3_norm = d3 / np.linalg.norm(d3)
    d4_norm = d4 / np.linalg.norm(d4)
    d5_norm = d5 / np.linalg.norm(d5)
    d6_norm = d6 / np.linalg.norm(d6)

    dx_norms = np.stack((d1_norm,
                         d2_norm,
                         d3_norm,
                         d4_norm,
                         d5_norm,
                         d6_norm))

    return dx_norms


def vib_analy(r, xyz, hessian):

    # r is the proton number of atoms
    # xyz is the cartesian coordinates in Angstrom
    # Hessian elements in atomic units (Ha/bohr^2)

    massvec = np.array([PT.GetAtomicWeight(i.item()) * AMU2KG
                        for i in list(np.array(r.reshape(-1)).astype(int))])
    expmassvec = np.repeat(massvec, 3)
    sqrtinvmassvec = np.divide(1.0, np.sqrt(expmassvec))
    hessian_mwc = np.einsum('i,ij,j->ij', sqrtinvmassvec,
                            hessian, sqrtinvmassvec)
    hessian_eigval, hessian_eigvec = np.linalg.eig(hessian_mwc)

    xyz_com, moi_eigvec = moi_tensor(massvec, expmassvec, xyz)
    dx_norms = trans_rot_vec(massvec, xyz_com, moi_eigvec)

    P = np.identity(3 * len(massvec))
    for dx_norm in dx_norms:
        P -= np.outer(dx_norm, dx_norm)

    # Projecting the T and R modes out of the hessian
    mwhess_proj = np.dot(P.T, hessian_mwc).dot(P)
    hess_proj = np.einsum('i,ij,j->ij', 1 / sqrtinvmassvec,
                          mwhess_proj, 1 / sqrtinvmassvec)

    hessian_eigval, hessian_eigvec = np.linalg.eigh(mwhess_proj)

    neg_ele = []
    for i, eigval in enumerate(hessian_eigval):
        if eigval < 0:
            neg_ele.append(i)

    hessian_eigval_abs = np.abs(hessian_eigval)

    pre_vib_freq_cm_1 = np.sqrt(
        hessian_eigval_abs * HA2J * 10e19) / (SPEEDOFLIGHT * 2 * np.pi *
                                              BOHRS2ANG * 100)

    vib_freq_cm_1 = pre_vib_freq_cm_1.copy()

    for i in neg_ele:
        vib_freq_cm_1[i] = -1 * pre_vib_freq_cm_1[i]

    trans_rot_elms = []
    for i, freq in enumerate(vib_freq_cm_1):
        # Modes that are less than 1.0 cm-1 are the
        # translation / rotation modes we just projected
        # out
        if np.abs(freq) < 1.0:
            trans_rot_elms.append(i)

    force_constants_J_m_2 = np.delete(
        hessian_eigval * HA2J * 1e20 / (BOHRS2ANG ** 2) * AMU2KG,
        trans_rot_elms)

    proj_vib_freq_cm_1 = np.delete(vib_freq_cm_1, trans_rot_elms)
    proj_hessian_eigvec = np.delete(hessian_eigvec.T, trans_rot_elms, 0)

    return (force_constants_J_m_2, proj_vib_freq_cm_1, proj_hessian_eigvec,
            mwhess_proj, hess_proj)


def free_rotor_moi(freqs):
    freq_ev = freqs * CM_TO_EV
    mu = 1 / (8 * np.pi ** 2 * freq_ev)
    return mu


def eff_moi(mu, b_av):
    mu_prime = mu * b_av / (mu + b_av)
    return mu_prime


def low_freq_entropy(freqs,
                     temperature,
                     b_av=B_AV):
    mu = free_rotor_moi(freqs)
    mu_prime = eff_moi(mu, b_av)

    arg = (8 * np.pi ** 3 * mu_prime * kB * temperature)
    entropy = GAS_CONST * (1 / 2 + np.log(arg ** 0.5))

    return entropy


def high_freq_entropy(freqs,
                      temperature):

    freq_ev = freqs * CM_TO_EV
    exp_pos = np.exp(freq_ev / (kB * temperature)) - 1
    exp_neg = 1 - np.exp(-freq_ev / (kB * temperature))

    entropy = GAS_CONST * (
        freq_ev / (kB * temperature * exp_pos) -
        np.log(exp_neg)
    )

    return entropy


def mrrho_entropy(freqs,
                  temperature,
                  rotor_cutoff,
                  b_av,
                  alpha):

    func = 1 / (1 + (rotor_cutoff / freqs) ** alpha)
    s_r = low_freq_entropy(freqs=freqs,
                           b_av=b_av,
                           temperature=temperature)
    s_v = high_freq_entropy(freqs=freqs,
                            temperature=temperature)

    new_vib_s = (func * s_v + (1 - func) * s_r).sum()
    old_vib_s = s_v.sum()

    return old_vib_s, new_vib_s


def mrrho_quants(ase_atoms,
                 freqs,
                 imag_cutoff=IMAG_CUTOFF,
                 temperature=TEMP,
                 pressure=PRESSURE,
                 rotor_cutoff=ROTOR_CUTOFF,
                 b_av=B_AV,
                 alpha=4,
                 flip_all_but_ts=False):

    potentialenergy = ase_atoms.get_potential_energy()

    if flip_all_but_ts:
        print(("Flipping all imaginary frequencies except "
               "the lowest one"))
        abs_freqs = abs(freqs[1:])

    else:
        abs_freqs = abs(freqs[freqs > imag_cutoff])
    ens = abs_freqs * CM_TO_EV

    ideal_gas = IdealGasThermo(vib_energies=ens,
                               potentialenergy=potentialenergy,
                               atoms=ase_atoms,
                               geometry='nonlinear',
                               symmetrynumber=1,
                               spin=0)

    # full entropy including rotation, translation etc
    old_entropy = (ideal_gas.get_entropy(temperature=temperature,
                                         pressure=pressure).item())
    enthalpy = (ideal_gas.get_enthalpy(temperature=temperature)
                .item())

    # correction to vibrational entropy
    out = mrrho_entropy(freqs=abs_freqs,
                        temperature=temperature,
                        rotor_cutoff=rotor_cutoff,
                        b_av=b_av,
                        alpha=alpha)
    old_vib_s, new_vib_s = out
    final_entropy = old_entropy - old_vib_s + new_vib_s

    free_energy = (enthalpy - temperature * final_entropy)

    return final_entropy, enthalpy, free_energy


def convert_modes(atoms,
                  modes):

    masses = (atoms.get_masses().reshape(-1, 1)
              .repeat(3, 1)
              .reshape(1, -1))

    # Multiply by 1 / sqrt(M) to be consistent with the DB
    vibdisps = modes / (masses ** 0.5)
    norm = np.linalg.norm(vibdisps, axis=1).reshape(-1, 1)

    # Normalize
    vibdisps /= norm

    # Re-shape

    num_atoms = len(atoms)
    vibdisps = vibdisps.reshape(-1, num_atoms, 3)

    return vibdisps


def hessian_and_modes(ase_atoms,
                      imag_cutoff=IMAG_CUTOFF,
                      rotor_cutoff=ROTOR_CUTOFF,
                      temperature=TEMP,
                      pressure=PRESSURE,
                      flip_all_but_ts=False,
                      analytical=False):

    # comparison to the analytical Hessian
    # shows that delta=0.005 is indistinguishable
    # from the real result, whereas delta=0.05
    # has up to 20% errors

    # delete the folder `vib` if it exists,
    # because it might mess up the Hessian
    # calculation

    if os.path.isdir('vib'):
        shutil.rmtree('vib')

    if analytical:
        hessian = analytical_hess(atoms=ase_atoms)

    else:
        vib = Vibrations(ase_atoms, delta=0.005)
        vib.run()

        vib_results = vib.get_vibrations()
        dim = len(ase_atoms)
        hessian = (vib_results.get_hessian()
                   .reshape(dim * 3, dim * 3) *
                   EV_TO_AU *
                   BOHR_RADIUS ** 2)

        print(vib.get_frequencies()[:20])

    vib_results = vib_analy(r=ase_atoms.get_atomic_numbers(),
                            xyz=ase_atoms.get_positions(),
                            hessian=hessian)
    _, freqs, modes, mwhess_proj, hess_proj = vib_results
    mwhess_proj *= AMU2KG

    vibdisps = convert_modes(atoms=ase_atoms,
                             modes=modes)

    mrrho_results = mrrho_quants(ase_atoms=ase_atoms,
                                 freqs=freqs,
                                 imag_cutoff=imag_cutoff,
                                 temperature=temperature,
                                 pressure=pressure,
                                 rotor_cutoff=rotor_cutoff,
                                 flip_all_but_ts=flip_all_but_ts)

    entropy, enthalpy, free_energy = mrrho_results

    imgfreq = len(freqs[freqs < 0])
    results = {"vibdisps": vibdisps.tolist(),
               "vibfreqs": freqs.tolist(),
               "modes": modes,
               "hessianmatrix": hessian.tolist(),
               "mwhess_proj": mwhess_proj.tolist(),
               "hess_proj": hess_proj.tolist(),
               "imgfreq": imgfreq,
               "freeenergy": free_energy * EV_TO_AU,
               "enthalpy": enthalpy * EV_TO_AU,
               "entropy": entropy * temperature * EV_TO_AU}

    return results

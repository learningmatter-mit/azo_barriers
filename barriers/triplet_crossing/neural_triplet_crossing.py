import os
import argparse
import numpy as np
import copy
from tqdm import tqdm
import pickle
import shutil

from ase import Atoms, optimize
from ase.calculators.calculator import Calculator, all_changes
from ase.vibrations import Vibrations
from nff.io.ase import NeuralFF, AtomsBatch
from nff.utils import constants as const
from nff.train import batch_to

from barriers.utils.neuraloptimizer import OPT_KEYS, get_opt_kwargs
from barriers.utils.vib import (vib_analy, convert_modes, mrrho_quants, AMU2KG, EV_TO_AU,
                                BOHR_RADIUS)
from barriers.irc.neural_irc import (take_step, make_en_func, get_model_path,
                                     coords_to_nxyz, get_modes, init_displace,
                                     load_info, load_model, convert_irc,
                                     back_convert_irc, COMPLETION_MESSAGE)

EV_TO_J = 1.602101e-19
ANGS_TO_M = 1e-10
KB = 1.380649e-23
H = 6.26e-34
HBAR = H / (2 * np.pi)
INV_CM_TO_J = 1.98630e-23


class GapCalc(Calculator):

    implemented_properties = ['energy', 'forces']

    def __init__(
        self,
        singlet_en_func,
        triplet_en_func,
        alpha_kcal,
        last_singlet_path=None,
        last_triplet_path=None,
        **kwargs
    ):

        Calculator.__init__(self, **kwargs)

        self.singlet_en_func = singlet_en_func
        self.triplet_en_func = triplet_en_func
        self.alpha_ev = alpha_kcal / const.EV_TO_KCAL_MOL

        self.last_singlet_path = last_singlet_path
        self.last_triplet_path = last_triplet_path

    def calculate(
        self,
        atoms=None,
        properties=['energy', 'forces'],
        system_changes=all_changes,
    ):

        # for backwards compatability
        if getattr(self, "properties", None) is None:
            self.properties = properties

        Calculator.calculate(self, atoms, self.properties, system_changes)

        # compute singlet and triplet energies and gradients
        nxyz = np.concatenate([atoms.get_atomic_numbers().reshape(-1, 1),
                               atoms.get_positions()], axis=-1)
        triplet_en, triplet_grad, last_triplet_path = self.triplet_en_func(
            nxyz=nxyz,
            grad=True,
            load_path=self.last_triplet_path,
            # don't need to convert coordinates, since they're already in Angstrom
            convert=False)

        singlet_en, singlet_grad, last_singlet_path = self.singlet_en_func(
            nxyz=nxyz,
            grad=True,
            load_path=self.last_singlet_path,
            # don't need to convert coordinates, since they're already in Angstrom
            convert=False)

        self.last_triplet_path = last_singlet_path
        self.last_singlet_path = last_singlet_path

        # our objective function is a smoothed-out version of the absolute
        # value of the gap, i.e. |gap|^2 / (|gap| + alpha), where alpha is a constant.
        # This comes from https://pubs.acs.org/doi/pdf/10.1021/jp0761618.
        # Their default of alpha = 0.02 Ha = 12.55 kcal/mol should work well

        # convert energies from Hartree to eV, and gradients from Hartree/Bohr
        # to eV/Angstrom

        triplet_en /= const.EV_TO_AU
        singlet_en /= const.EV_TO_AU

        triplet_grad /= (const.EV_TO_AU * const.BOHR_RADIUS)
        singlet_grad /= (const.EV_TO_AU * const.BOHR_RADIUS)

        # objective function
        gap = (singlet_en - triplet_en)
        alpha = self.alpha_ev
        obj = gap ** 2 / (abs(gap) + alpha)

        # gradient of objective function

        gap_grad = (singlet_grad - triplet_grad)
        num = (gap ** 3 + 2 * alpha * gap * abs(gap)) * gap_grad
        denom = abs(gap) * (alpha + abs(gap)) ** 2
        obj_grad = num / denom

        self.results = {
            'energy': obj.reshape(-1),
            'forces': -obj_grad.reshape(-1, 3)
        }


class ProjectedEnergyCalc(Calculator):

    """
    Calculator for minimizing the average energy of the singlet and triplet
    states, while projecting out the component of the force in the direction
    of grad (E_singlet - E_triplet). This means that if you start at a crossing
    point between singlet and triplet, you should stay at a crossing point, but
    lower the energy of each.
    """

    implemented_properties = ['energy', 'forces']

    def __init__(self,
                 singlet_model,
                 triplet_model,
                 alpha_kcal=12.55,
                 **kwargs):

        Calculator.__init__(self, **kwargs)
        self.singlet_calc = self.get_singlet_calc(singlet_model,
                                                  **kwargs)
        self.triplet_calc = self.get_triplet_calc(triplet_model,
                                                  **kwargs)
        self.alpha = alpha_kcal / const.EV_TO_KCAL_MOL

    def get_singlet_calc(self,
                         singlet_model,
                         **kwargs):

        singlet_kwargs = copy.deepcopy(kwargs)
        singlet_kwargs.update({"model": singlet_model})
        calc = NeuralFF(**singlet_kwargs)

        return calc

    def get_triplet_calc(self,
                         triplet_model,
                         **kwargs):

        en_key = kwargs.get("en_key", "energy")
        grad_key = "%s_grad" % en_key
        grad_keys = [grad_key]

        if hasattr(triplet_model, "grad_keys"):
            triplet_model.grad_keys = grad_keys

        elif hasattr(triplet_model, "painn_model"):
            triplet_model.painn_model.grad_keys = grad_keys
        else:
            msg = "Don't know how to set the grad keys of the triplet model"
            raise Exception(msg)

        triplet_kwargs = copy.deepcopy(kwargs)
        triplet_kwargs.update({"model": triplet_model})
        calc = NeuralFF(**triplet_kwargs)

        return calc

    def calculate(
            self,
            atoms=None,
            properties=['energy', 'forces'],
            system_changes=all_changes,
    ):

        batch = atoms.get_batch()
        assert len(batch['num_atoms']) == 1, "Not set up for batching"

        self.singlet_calc.calculate(atoms=atoms,
                                    properties=properties,
                                    system_changes=system_changes)

        self.triplet_calc.calculate(atoms=atoms,
                                    properties=properties,
                                    system_changes=system_changes)

        singlet_props = self.singlet_calc.results
        triplet_props = self.triplet_calc.results

        mean_en = 1 / 2 * (singlet_props["energy"] + triplet_props["energy"])
        mean_f = 1 / 2 * (singlet_props["forces"] + triplet_props["forces"])

        delta_en = singlet_props["energy"] - triplet_props["energy"]
        delta_f = singlet_props["forces"] - triplet_props["forces"]

        unit_df = delta_f / np.linalg.norm(delta_f)
        proj_df = (unit_df * mean_f).sum()
        use_f = (mean_f - proj_df * unit_df +
                 (delta_en / self.alpha) * delta_f)

        self.results.update({"energy": mean_en,
                             "forces": use_f})


def make_eff_hess(singlet_f,
                  triplet_f,
                  singlet_h,
                  triplet_h):

    g_1 = singlet_f.reshape(-1)
    g_2 = triplet_f.reshape(-1)

    dim = g_1.shape[0]
    h_1 = singlet_h.reshape(dim, dim)
    h_2 = triplet_h.reshape(dim, dim)

    dot = (g_1 * g_2).sum()
    sign = -1 if dot > 0 else 1

    norm_g_1 = np.linalg.norm(g_1)
    norm_g_2 = np.linalg.norm(g_2)
    norm_dg = np.linalg.norm(g_1 - g_2)

    h_eff = (norm_g_1 * h_2 + sign * norm_g_2 * h_1) / norm_dg

    n = g_1 - g_2
    inner_n = np.linalg.norm(n) ** 2
    h_prime = (h_eff - np.outer(np.einsum('ij, j-> i', h_eff, n), n) / inner_n -
               np.outer(n, np.einsum('ij, j-> i', h_eff, n)) / inner_n +
               (n * np.einsum('ij, j->i', h_eff, n)).sum() *
               np.outer(n, n) / inner_n ** 2
               )

    return h_prime


def get_forces(atoms_batch,
               proj_energy_calc):

    force_list = []
    for name in ["singlet", "triplet"]:
        calc = getattr(proj_energy_calc, "%s_calc" % name)
        atoms_batch.set_calculator(calc)
        force_list.append(atoms_batch.get_forces())

    return force_list


def get_hessians(atoms_batch,
                 proj_energy_calc):

    hessians = []

    for name in ["singlet", "triplet"]:

        if os.path.isdir('vib'):
            shutil.rmtree('vib')

        calc = getattr(proj_energy_calc, "%s_calc" % name)
        atoms_batch.set_calculator(calc)

        vib = Vibrations(atoms_batch, delta=0.005)
        vib.run()

        vib_results = vib.get_vibrations()
        dim = len(atoms_batch)
        hessian = (vib_results.get_hessian()
                   .reshape(dim * 3, dim * 3) *
                   EV_TO_AU *
                   BOHR_RADIUS ** 2)

        hessians.append(hessian)

        if os.path.isdir('vib'):
            shutil.rmtree('vib')

    return hessians


def do_hess_analysis(atoms_batch,
                     singlet_f,
                     triplet_f,
                     singlet_h,
                     triplet_h,
                     imag_cutoff,
                     rotor_cutoff,
                     temperature,
                     pressure,
                     flip_all_but_ts):

    h_prime = make_eff_hess(singlet_f=singlet_f,
                            triplet_f=triplet_f,
                            singlet_h=singlet_h,
                            triplet_h=triplet_h)

    vib_results = vib_analy(r=atoms_batch.get_atomic_numbers(),
                            xyz=atoms_batch.get_positions(),
                            hessian=h_prime)
    _, freqs, modes, mwhess_proj, hess_proj = vib_results
    mwhess_proj *= AMU2KG

    vibdisps = convert_modes(atoms=atoms_batch,
                             modes=modes)

    mrrho_results = mrrho_quants(ase_atoms=atoms_batch,
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
               "hessianmatrix": h_prime.tolist(),
               "mwhess_proj": mwhess_proj.tolist(),
               "hess_proj": hess_proj.tolist(),
               "imgfreq": imgfreq,
               "freeenergy": free_energy * EV_TO_AU,
               "enthalpy": enthalpy * EV_TO_AU,
               "entropy": entropy * temperature * EV_TO_AU}

    return results


def hess_and_k_isc(atoms_batch,
                   proj_calc,
                   params):

    force_list = get_forces(atoms_batch=atoms_batch,
                            proj_energy_calc=proj_calc)
    k_isc = get_k_isc(atoms_batch=atoms_batch,
                      singlet_f=force_list[0],
                      triplet_f=force_list[1],
                      params=params)

    if params["compute_hessian"]:
        hessians = get_hessians(atoms_batch=atoms_batch,
                                proj_energy_calc=proj_calc)
        results = do_hess_analysis(atoms_batch=atoms_batch,
                                   singlet_f=force_list[0],
                                   triplet_f=force_list[1],
                                   singlet_h=hessians[0],
                                   triplet_h=hessians[1],
                                   imag_cutoff=params['imag_cutoff'],
                                   rotor_cutoff=params['rotor_cutoff'],
                                   temperature=params['temperature'],
                                   pressure=params['pressure'],
                                   flip_all_but_ts=False)
    else:
        results = None

    return results, k_isc


def wkb_analytical(atoms_batch,
                   h_so_inv_cm,
                   singlet_f,
                   triplet_f,
                   temperature):

    # Analytical approximation to the ISC rate from WKB theory
    # https://doi.org/10.1021/jp503794j

    f_conv = (EV_TO_J / ANGS_TO_M)

    # forces originally in ase units (eV / A)
    f = (abs((singlet_f * triplet_f).sum()) ** 0.5) * f_conv
    df = np.linalg.norm(singlet_f - triplet_f) * f_conv

    df_dir = (singlet_f - triplet_f) / np.linalg.norm(singlet_f - triplet_f)
    massvec = atoms_batch.get_masses() * AMU2KG
    mu = 1 / (df_dir ** 2 / massvec.reshape(-1, 1)).sum()

    h_so = h_so_inv_cm * INV_CM_TO_J
    kt = KB * temperature
    beta = 4 * h_so ** (3 / 2) / HBAR * (mu / (f * df)) ** 0.5

    e0 = df / (2 * f * h_so)
    gamma = (np.pi ** (3 / 2) * beta) / (2 * (float(e0) / float(kt)) ** 0.5
                                         ) * (1 + 1 / 2 * np.exp(
                                             1 / (12 * beta ** 2) *
                                             1 / (kt * e0) ** 3
                                         ))
    # the rate is gamma / h, not gamma / (kT h) [there's a typo in the paper]

    k_isc = gamma / H

    return k_isc


def get_k_isc(atoms_batch,
              singlet_f,
              triplet_f,
              params):

    k_isc_params = params['k_isc_params']
    h_so_inv_cm = k_isc_params["h_so_inv_cm"]
    method = k_isc_params["method"]

    if method == "wkb_analytical":
        k_isc = wkb_analytical(atoms_batch=atoms_batch,
                               h_so_inv_cm=h_so_inv_cm,
                               singlet_f=singlet_f,
                               triplet_f=triplet_f,
                               temperature=params['temperature'])
    else:
        raise Exception(("Intersystem crossing rate not implemented "
                         "for %s" % method))

    # note - don't use factor of two or three because we don't know if it's
    # on the reactant side or product side

    return k_isc


def get_opt_module(params,
                   atoms_batch):
    nbr_update_period = params["nbr_list_update_freq"]
    opt_kwargs = get_opt_kwargs(params=params,
                                nbr_update_period=nbr_update_period)
    opt_name = params["opt_type"]
    opt_module = getattr(optimize, opt_name)

    dyn = opt_module(atoms_batch)
    max_steps = params["opt_max_step"]

    return dyn, opt_kwargs, max_steps, nbr_update_period


def further_opt(params,
                atoms_batch):
    """
    Once you've hit a singlet-triplet crossing, further refine it by minimizing
    the average singlet-triplet energy, while keeping them as close as possible
    to equal
    """

    singlet_params = params['singlet_params']
    triplet_params = params['triplet_params']

    models = []
    for these_params in [singlet_params, triplet_params]:
        model_path = get_model_path(these_params)
        model = load_model(model_path)
        models.append(model)

    singlet_model, triplet_model = models
    device = params["device"]
    proj_calc = ProjectedEnergyCalc(en_key='energy_0',
                                    device=device,
                                    singlet_model=singlet_model,
                                    triplet_model=triplet_model,
                                    alpha_kcal=params["alpha_kcal"])
    atoms_batch.set_calculator(proj_calc)
    out = get_opt_module(params=params,
                         atoms_batch=atoms_batch)
    dyn, opt_kwargs, max_steps, nbr_update_period = out

    total_steps = 0
    dyn_converged = False

    while total_steps < max_steps:
        dyn_converged = dyn.run(**opt_kwargs)

        if dyn_converged:
            break

        atoms_batch.update_nbr_list()
        opt_kwargs['steps'] += nbr_update_period
        total_steps += nbr_update_period

    batch = batch_to(atoms_batch.get_batch(), device)
    singlet_results = singlet_model(batch)
    triplet_results = triplet_model(batch)

    singlet_en = float(singlet_results[params['en_key']] *
                       const.KCAL_TO_AU['energy'])
    triplet_en = float(triplet_results[params['en_key']] *
                       const.KCAL_TO_AU['energy'])

    mode_results, k_isc = hess_and_k_isc(atoms_batch=atoms_batch,
                                         proj_calc=proj_calc,
                                         params=params)

    return singlet_en, triplet_en, mode_results, k_isc


def refine_crossings(trj,
                     params):

    print(("Minimizing average (singlet, triplet) energy while keeping "
           "the gap as small as possible"))

    atoms_batch = AtomsBatch(numbers=trj[-2][:, 0],
                             positions=trj[-1][:, 1:],
                             cutoff=params["cutoff"],
                             cutoff_skin=params["cutoff_skin"],
                             directed=params["directed"],
                             requires_large_offsets=params["requires_large_offsets"],
                             device=params["device"])

    singlet_en, triplet_en, mode_results, k_isc = further_opt(params=params,
                                                              atoms_batch=atoms_batch)
    opt_atoms = Atoms(numbers=atoms_batch.get_atomic_numbers(),
                      positions=atoms_batch.get_positions())

    return opt_atoms, singlet_en, triplet_en, mode_results, k_isc


def run_until_crossing(g0_nxyz,
                       en_func,
                       scale_displ_sd,
                       sd_parabolic_fit,
                       interpolate_only,
                       do_sd_corr,
                       sd_corr_parabolic_fit,
                       scale_displ_sd_corr,
                       tol_rms_g,
                       tol_max_g,
                       max_steps,
                       adapt_scale_disp,
                       triplet_en_func,
                       **kwargs):

    trj = [g0_nxyz]
    singlet_ens = []
    triplet_ens = []

    init_scale_displ_sd = copy.deepcopy(scale_displ_sd)
    last_path = None
    last_triplet_path = None
    found_triplet = False

    for num_steps in tqdm(range(max_steps)):

        out = take_step(
            g0_nxyz=g0_nxyz,
            en_func=en_func,
            scale_displ_sd=scale_displ_sd,
            sd_parabolic_fit=sd_parabolic_fit,
            interpolate_only=interpolate_only,
            do_sd_corr=do_sd_corr,
            sd_corr_parabolic_fit=sd_corr_parabolic_fit,
            scale_displ_sd_corr=scale_displ_sd_corr,
            last_path=last_path,
            adapt_scale_disp=adapt_scale_disp,
            init_scale_displ_sd=init_scale_displ_sd
        )

        g1_nxyz, last_path, grad_0, singlet_en, scale_displ_sd = out
        singlet_ens.append(float(singlet_en))

        triplet_en, _, last_triplet_path = triplet_en_func(nxyz=g1_nxyz,
                                                           grad=False,
                                                           load_path=last_triplet_path)

        triplet_ens.append(float(triplet_en))

        if len(triplet_ens) >= 2:
            old_triplet_en = triplet_ens[-2]
            old_singlet_en = singlet_ens[-2]
            old_sign = np.sign(old_triplet_en - old_singlet_en)
            new_sign = np.sign(triplet_en - singlet_en)

            if old_sign != new_sign:
                found_triplet = True

        if found_triplet or num_steps == (max_steps - 1):
            if found_triplet:
                print("Found a triplet crossing after %d steps!" %
                      (num_steps + 1))
            break

        trj.append(copy.deepcopy(g1_nxyz))
        g0_nxyz = g1_nxyz

    if not found_triplet:
        print("Did not find a triplet crossing after %d steps" % (num_steps + 1))

    # convert trajectory to Angstrom
    trj = convert_irc(trj)

    return trj, singlet_ens, triplet_ens, found_triplet


def implicit_opt_gap(trj,
                     params,
                     en_func,
                     triplet_en_func):

    print("Re-running with smaller steps to get exact crossing...")
    remain_params = {key: val for key, val in params.items()
                     if key not in ['scale_displ_sd', 'max_steps']}

    new_trj, singlet_ens, triplet_ens, _ = run_until_crossing(
        g0_nxyz=back_convert_irc([trj[-2]])[0],
        scale_displ_sd=params["crossing_scale_displ_sd"],
        max_steps=params["crossing_max_steps"],
        en_func=en_func,
        triplet_en_func=triplet_en_func,
        **remain_params)

    gap = abs(np.array(singlet_ens) - np.array(triplet_ens))
    argmin = np.argmin(gap)
    last_xyz = new_trj[argmin]

    opt_atoms = Atoms(numbers=last_xyz[:, 0],
                      positions=last_xyz[:, 1:])
    singlet_en = singlet_ens[-1]
    triplet_en = triplet_ens[-1]

    return opt_atoms, singlet_en, triplet_en


def run_opt(params, atoms):
    opt_kwargs = {key: val for key,
                  val in params.items() if key in OPT_KEYS}
    opt_kwargs.update({"steps": params.get("opt_max_step", 500)})

    opt_module = getattr(optimize, params.get("opt_type", "BFGS"))
    dyn = opt_module(atoms)

    # no need to split into multiple sections, each of which is followed by a
    # neighbor list update, because `en_func` does a neighbor list update for
    # every new geometry
    dyn.run(**opt_kwargs)


def combined_calc(atoms,
                  singlet_en_func,
                  triplet_en_func):

    nxyz = np.concatenate([atoms.get_atomic_numbers().reshape(-1, 1),
                           atoms.get_positions()], axis=-1)

    triplet_en, _, _ = triplet_en_func(
        nxyz=nxyz,
        grad=False,
        load_path=None,
        # don't need to convert coordinates, since they're already in Angstrom
        convert=False)

    singlet_en, _, _ = singlet_en_func(
        nxyz=nxyz,
        grad=False,
        load_path=None,
        # don't need to convert coordinates, since they're already in Angstrom
        convert=False)

    return float(singlet_en), float(triplet_en)


def opt_gap(trj,
            singlet_ens,
            triplet_ens,
            singlet_en_func,
            triplet_en_func,
            alpha_kcal,
            params):
    """
    Find the geometry where the singlet-triplet gap is zero
    """

    gaps = abs(np.array(singlet_ens) - np.array(triplet_ens))
    # our initial guess
    start_xyz = trj[np.argmin(gaps)]
    # our calculator to output a smoothed-out version of |gap|, and the associated
    # gradient
    gap_calc = GapCalc(singlet_en_func=singlet_en_func,
                       triplet_en_func=triplet_en_func,
                       alpha_kcal=alpha_kcal,
                       last_singlet_path=None,
                       last_triplet_path=None)

    # make an atoms object
    atoms = Atoms(numbers=start_xyz[:, 0],
                  positions=start_xyz[:, 1:])
    atoms.set_calculator(gap_calc)

    # use an ASE optimizer to minimize the absolute value of the gap
    run_opt(params=params,
            atoms=atoms)

    # return the atoms and the singlet/triplet energies in AU
    singlet_en, triplet_en = combined_calc(atoms=atoms,
                                           singlet_en_func=singlet_en_func,
                                           triplet_en_func=triplet_en_func)

    return atoms, singlet_en, triplet_en


def make_en_funcs(params, job_dir):

    singlet_params = params['singlet_params']
    triplet_params = params['triplet_params']

    en_funcs = []
    for these_params in [singlet_params, triplet_params]:
        model_path = get_model_path(these_params)

        # make sure it gets the gradient too

        device = params['device']
        en_key = these_params['en_key']

        model = load_model(model_path)
        model = model.to(device)

        grad_key = en_key + "_grad"
        if hasattr(model, "grad_keys"):
            grad_keys = model.grad_keys
        elif hasattr(model, "painn_model"):
            grad_keys = model.painn_model.grad_keys
        else:
            raise Exception("Can't find grad keys!")

        if grad_key not in grad_keys:
            grad_keys += [grad_key]

        en_func = make_en_func(calc_type='neural',
                               neural_en_key=en_key,
                               cutoff=these_params['cutoff'],
                               device=device,
                               model_path=None,
                               config_name=None,
                               jobspec=None,
                               job_dir=job_dir,
                               grad_key=None,
                               model_kwargs=these_params.get('model_kwargs'),
                               model=model)
        en_funcs.append(en_func)

    singlet_en_func, triplet_en_func = en_funcs

    return singlet_en_func, triplet_en_func


def get_mode_info(params,
                  job_dir):

    ts_nxyz = coords_to_nxyz(params['coords'])

    singlet_params = params['singlet_params']
    model_path = get_model_path(singlet_params)

    # do a Hessian calculation

    device = params["device"]
    cutoff = singlet_params["cutoff"]
    neural_en_key = singlet_params["en_key"]
    model_kwargs = singlet_params.get("model_kwargs")

    eigvecs, freqs_cm = get_modes(job_dir=job_dir,
                                  ts_nxyz=ts_nxyz,
                                  model_path=model_path,
                                  neural_en_key=neural_en_key,
                                  device=device,
                                  cutoff=cutoff,
                                  model_kwargs=model_kwargs)

    return eigvecs, freqs_cm, ts_nxyz


def xyz_list_to_trj(xyz_list):
    trj = [Atoms(numbers=nxyz[:, 0],
                 positions=nxyz[:, 1:])
           for nxyz in xyz_list]

    return trj


def reorder_irc(ircs,
                ts_nxyz,
                singlet_ens,
                triplet_ens,
                singlet_en_func,
                triplet_en_func):

    final_irc = list(reversed(ircs[0]))
    final_irc.append(ts_nxyz)
    final_irc += ircs[1]

    singlet_ts_en, _, _ = singlet_en_func(nxyz=ts_nxyz,
                                          grad=False,
                                          load_path=None,
                                          convert=False)

    triplet_ts_en, _, _ = triplet_en_func(nxyz=ts_nxyz,
                                          grad=False,
                                          load_path=None,
                                          convert=False)

    ts_ens = [singlet_ts_en, triplet_ts_en]
    en_set = [singlet_ens, triplet_ens]

    all_final_ens = []
    for i, ens in enumerate(en_set):
        final_ens = list(reversed(ens[0]))
        final_ens.append(float(ts_ens[i]))
        final_ens += ens[1]

        all_final_ens.append(final_ens)

    final_singlet_ens, triplet_ens = all_final_ens

    return final_irc, final_singlet_ens, triplet_ens


def run_all(params, job_dir):
    eigvecs, freqs_cm, ts_nxyz = get_mode_info(params=params,
                                               job_dir=job_dir)
    nxyz_fwd, nxyz_bckwd = init_displace(eigvecs=eigvecs,
                                         freqs_cm=freqs_cm,
                                         init_displ_de=params["init_displ_de"],
                                         ts_nxyz=ts_nxyz,
                                         mode=-1)

    singlet_en_func, triplet_en_func = make_en_funcs(params=params,
                                                     job_dir=job_dir)

    summary = {'path': {},
               'opt': {'atoms': [],
                       'singlet_energies': [],
                       'triplet_energies': [],
                       'mode_results': [],
                       'k_isc': []}}

    ircs = []
    all_singlet_ens = []
    all_triplet_ens = []

    for i, g0_nxyz in enumerate([nxyz_fwd, nxyz_bckwd]):
        key = 'forward' if i == 0 else 'backward'
        print("Generating %s path to find %s triplet crossing" % (key, key))

        trj, singlet_ens, triplet_ens, found_triplet = run_until_crossing(
            g0_nxyz=g0_nxyz,
            en_func=singlet_en_func,
            scale_displ_sd=params["scale_displ_sd"],
            sd_parabolic_fit=params["sd_parabolic_fit"],
            interpolate_only=params["interpolate_only"],
            do_sd_corr=params["do_sd_corr"],
            sd_corr_parabolic_fit=params["sd_corr_parabolic_fit"],
            scale_displ_sd_corr=params["scale_displ_sd_corr"],
            tol_rms_g=params["tol_rms_g"],
            tol_max_g=params["tol_max_g"],
            max_steps=params["max_iter"],
            adapt_scale_disp=params["adapt_scale_disp"],
            triplet_en_func=triplet_en_func)

        if found_triplet:
            out = refine_crossings(trj=trj,
                                   params=params)
            opt_atoms, singlet_en, triplet_en, mode_results, k_isc = out

        else:
            opt_atoms = Atoms(numbers=trj[-1][:, 0],
                              positions=trj[-1][:, 1:])
            singlet_en = singlet_ens[-1]
            triplet_en = triplet_ens[-1]
            mode_results = None
            k_isc = None

        ircs.append(trj)
        all_singlet_ens.append(singlet_ens)
        all_triplet_ens.append(triplet_ens)

        summary['opt']['atoms'].append(opt_atoms)
        summary['opt']['singlet_energies'].append(singlet_en)
        summary['opt']['triplet_energies'].append(triplet_en)
        summary['opt']['mode_results'].append(mode_results)
        summary['opt']['k_isc'].append(k_isc)

    final_irc, final_singlet_ens, triplet_ens = reorder_irc(
        ircs=ircs,
        ts_nxyz=ts_nxyz,
        singlet_ens=all_singlet_ens,
        triplet_ens=all_triplet_ens,
        singlet_en_func=singlet_en_func,
        triplet_en_func=triplet_en_func)

    summary['path'] = {"atoms": xyz_list_to_trj(final_irc),
                       "singlet_energies": final_singlet_ens,
                       "triplet_energies": triplet_ens}

    return summary


def main(job_dir, info_path):
    params = load_info(info_path)
    summary = run_all(params, job_dir)
    save_path = os.path.join(job_dir, 'triplet_opt.pickle')
    with open(save_path, 'wb') as f:
        pickle.dump(summary, f)

    print(COMPLETION_MESSAGE)


def run_from_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_dir',
                        type=str,
                        default='.',
                        help=("Where to do the calculation"))
    parser.add_argument('--config_file',
                        type=str,
                        default='job_info.json',
                        help=("Name of the job info file"))

    args = parser.parse_args()
    info_path = os.path.join(args.job_dir, args.config_file)

    try:
        main(job_dir=args.job_dir,
             info_path=info_path)
    except Exception as e:
        print(e)
        import pdb
        pdb.post_mortem()


if __name__ == "__main__":
    run_from_command_line()

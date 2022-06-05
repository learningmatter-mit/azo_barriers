import numpy as np
import pickle
import json
from rdkit import Chem
import copy
from jinja2 import Template
import importlib
import os
from tqdm import tqdm
import argparse
from ase import Atoms
import shutil
from torch.utils.data import DataLoader

from nff.utils import constants as const
from nff.utils.misc import bash_command
from nff.train import load_model, batch_to, batch_detach
from nff.data import collate_dicts, Dataset
from nff.io.ase import UNDIRECTED, NeuralFF, AtomsBatch

from barriers.utils.vib import hessian_and_modes

PERIODICTABLE = Chem.GetPeriodicTable()
EPS = 1e-15

CONFIG_PATHS = {"bhhlyp_6-31gs_sf_tddft_engrad_qchem":
                "qchem/bhhlyp_6-31gs_sf_tddft_engrad"}

COMPLETION_MESSAGE = 'Neural IRC terminated normally.'


def get_dx(init_displ_de,
           freq_cm,
           eigvec,
           masses):
    """
    Get coordinate diplacement.
    Args:
    init_displ_de (float): target energy change, in a.u.
    freq_cm (float): frequency in cm^(-1)
    eigvec (np.array): normalized eigenvector of the mass-weighted Hessian
    with the lowest (most negative) frequency
    masses (np.array): masses, in atomic units
    (units of m_e, the electron mass)
    """

    # squared frequency, in a.u.
    w2_au = (freq_cm * const.INV_CM_TO_AU) ** 2

    # `scale` has units of sqrt(m_e) * Bohr,
    # where m_e is the electron mass
    scale = (2 * np.abs(init_displ_de) /
             np.abs(w2_au)) ** 0.5

    # displacement in mass-weighted Cartesian coordinates,
    # in a.u. (Bohr radii)
    dx_tilde = scale * eigvec / np.linalg.norm(eigvec)

    # displacement in Cartesian coordinates, in Bohr radii

    dx_bohr = (dx_tilde.reshape(-1, 3) /
               (masses.reshape(-1, 1) ** 0.5))

    return dx_bohr


def get_masses(nxyz):
    """
    Args:
    nxyz (np.array): xyz with atomic numbers
    """

    # masses in atomic mass units
    m_amu = np.array([PERIODICTABLE.GetAtomicWeight(int(i))
                      for i in nxyz[:, 0]])

    # masses in atomic units

    masses = m_amu * const.AMU_TO_AU

    return masses


def init_displace(eigvecs,
                  freqs_cm,
                  init_displ_de,
                  ts_nxyz,
                  mode=-1):
    """
    All inputs are in atomic units
    """

    # get eigenvalue and eigenvector with
    # the lowest frequency

    if mode == -1:
        argmin = np.argmin(freqs_cm)
    else:
        argmin = mode

    eigvec = eigvecs[argmin]
    freq_cm = freqs_cm[argmin]

    # get the displacement in the direction of
    # `eigvec` that will give an energy change of
    # `init_displ_de`

    masses = get_masses(nxyz=ts_nxyz)

    # displacement in Bohrs
    d_xyz = get_dx(init_displ_de=init_displ_de,
                   freq_cm=freq_cm,
                   eigvec=eigvec,
                   masses=masses)

    signs = [1, -1]
    new_nxyzs = []
    for sign in signs:
        new_nxyz = copy.deepcopy(ts_nxyz)
        # convert to Bohr
        new_nxyz[:, 1:] /= const.BOHR_RADIUS
        new_nxyz[:, 1:] += sign * d_xyz
        new_nxyzs.append(new_nxyz)

    nxyz_fwd, nxyz_bckwd = new_nxyzs

    return nxyz_fwd, nxyz_bckwd


def init_sd_step(g0_nxyz,
                 grad_0,
                 scale_displ_sd):
    """
    Initial steepest descent step. A separate function
    adjusts the step if the energy increases after it.

    Args:
    g0_nxyz(np.array): nxyz at the starting geometry
    grad_0(np.array): gradient at the starting geometry
    (after the initial Hessian displacement)
    scale_displ_sd(float): scaling factor for the gradient

    """

    sd_1 = -scale_displ_sd * (grad_0 / np.linalg.norm(grad_0))
    g1_nxyz = copy.deepcopy(g0_nxyz)
    g1_nxyz[:, 1:] += sd_1

    return g1_nxyz, sd_1


def add_sd_2(g0_nxyz,
             g1_nxyz,
             sd_1):
    """
    Generation of `sd_2` if g1 (=g0 + sd_1)
    has a higher energy than g0.
    """

    sd_2 = sd_1 * 0.5
    g2_nxyz = copy.deepcopy(g0_nxyz)
    g2_nxyz[:, 1:] += sd_2

    return g2_nxyz


def best_step_quadratic(e0,
                        e1,
                        e2):
    """
    e0, e1, and e2 are the energies in
    order *of increasing step size*. That means
    that, if you take a step from g0 to g2,
    and then add a step that's half the size,
    you get e0 = e(g0), e1 = e(g2), and
    e2 = e(g1). Don't let that mess things up!
    """

    a = -4 * (e1 - e0 / 2 - e2 / 2)
    b = -3 * e0 + 4 * e1 - e2

    # check curvature - if a < 0
    # then the optimal energies are
    # at the end points

    if a > 0:
        best_step = -b / (2 * a + EPS)
    else:
        best_step = 0 if (e0 < e2) else 1

    return best_step


def sd_parabolic(g0_nxyz,
                 g1_nxyz,
                 g2_nxyz,
                 g0_en,
                 g1_en,
                 g2_en,
                 sd_1,
                 interpolate_only):
    """
    Do a parabolic interpolation of the energies
    at g0, g1 and g2 to find the best step.
    """

    best_step = best_step_quadratic(e0=g0_en,
                                    e1=g2_en,
                                    e2=g1_en)

    # if `interpolate_only` is set to True, then
    # you can only take step within the confines
    # of g0 and g1

    if interpolate_only:
        if best_step < 0:
            best_step = 0
        elif best_step > 1:
            best_step = 1

    xyz_displace = sd_1 * best_step
    gfinal_nxyz = copy.deepcopy(g0_nxyz)
    gfinal_nxyz[:, 1:] += xyz_displace

    return gfinal_nxyz


def sd_corr(grad_0,
            grad_1,
            g0_nxyz,
            g1_nxyz,
            scale_displ_sd_corr):
    """
    Correction to steepest descent step
    """

    # corrected direction

    d = (grad_0 / np.linalg.norm(grad_0) -
         grad_1 / np.linalg.norm(grad_1))
    d /= np.linalg.norm(d)

    # change in coordinates from steepest descent guess
    d_xyz = g1_nxyz[:, 1:] - g0_nxyz[:, 1:]

    sdc_1 = (d * scale_displ_sd_corr *
             np.linalg.norm(d_xyz))

    sdc_1_nxyz = copy.deepcopy(g1_nxyz)
    sdc_1_nxyz[:, 1:] += sdc_1

    return sdc_1_nxyz, sdc_1


def sd_corr_parabolic_step(sdc_1_nxyz,
                           sdc_1,
                           g1_en,
                           sdc_1_en,
                           g1_nxyz):
    """
    Parabolic correction to SD correction, whether
    the energy increases or decreases
    """

    increased = (sdc_1_en > g1_en)
    factor = 0.5 if increased else 2
    sdc_2 = sdc_1 * factor

    sdc_2_nxyz = copy.deepcopy(g1_nxyz)
    sdc_2_nxyz[:, 1:] += sdc_2

    return sdc_2_nxyz


def sd_corr_parabolic(g1_nxyz,
                      sdc_1_nxyz,
                      sdc_2_nxyz,
                      g1_en,
                      sdc_1_en,
                      sdc_2_en,
                      sdc_1,
                      interpolate_only):

    e0 = g1_en
    increased = (sdc_1_en > g1_en)
    if increased:
        e1 = sdc_2_en
        e2 = sdc_1_en
        max_val = 1
    else:
        e1 = sdc_1_en
        e2 = sdc_2_en
        max_val = 2

    best_step = best_step_quadratic(e0=e0,
                                    e1=e1,
                                    e2=e2)

    # if `interpolate_only` is set to True, then
    # you can only take step within the confines
    # of g0 and g1

    if interpolate_only:
        if best_step < 0:
            best_step = 0
        elif best_step > 1:
            best_step = 1

    xyz_displace = sdc_1 * best_step * max_val
    gfinal_final_nxyz = copy.deepcopy(g1_nxyz)
    gfinal_final_nxyz[:, 1:] += xyz_displace

    return gfinal_final_nxyz


def sd_before_correction(g0_nxyz,
                         en_func,
                         interpolate_only,
                         scale_displ_sd,
                         sd_parabolic_fit,
                         last_path):

    g0_en, grad_0, last_path = en_func(nxyz=g0_nxyz,
                                       grad=True,
                                       load_path=last_path)

    g1_nxyz, sd_1 = init_sd_step(g0_nxyz=g0_nxyz,
                                 grad_0=grad_0,
                                 scale_displ_sd=scale_displ_sd)

    if not sd_parabolic_fit:
        return g1_nxyz, last_path, grad_0, g0_en

    g1_en, _, last_path = en_func(nxyz=g1_nxyz,
                                  grad=False,
                                  load_path=last_path)

    if g1_en < g0_en:
        return g1_nxyz, last_path, grad_0, g0_en

    g2_nxyz = add_sd_2(g0_nxyz=g0_nxyz,
                       g1_nxyz=g1_nxyz,
                       sd_1=sd_1)

    g2_en, _, last_path = en_func(nxyz=g2_nxyz,
                                  grad=False,
                                  load_path=last_path)

    gfinal_nxyz = sd_parabolic(g0_nxyz=g0_nxyz,
                               g1_nxyz=g1_nxyz,
                               g2_nxyz=g2_nxyz,
                               g0_en=g0_en,
                               g1_en=g1_en,
                               g2_en=g2_en,
                               sd_1=sd_1,
                               interpolate_only=interpolate_only)

    return gfinal_nxyz, last_path, grad_0, g0_en


def do_scale_adapt(g0_nxyz,
                   gfinal_nxyz,
                   scale_displ_sd,
                   init_scale_displ_sd):

    old_xyz = g0_nxyz[:, 1:]
    new_xyz = gfinal_nxyz[:, 1:]
    actual_displ = np.linalg.norm(new_xyz - old_xyz)

    if actual_displ <= 0.5 * scale_displ_sd:
        scale_displ_sd *= 0.5

    elif actual_displ >= 2 * scale_displ_sd:
        scale_displ_sd *= 2

    if scale_displ_sd >= 4 * init_scale_displ_sd:
        scale_displ_sd = 4 * init_scale_displ_sd

    elif scale_displ_sd <= init_scale_displ_sd / 16:
        scale_displ_sd = init_scale_displ_sd / 16

    return scale_displ_sd


def take_step(g0_nxyz,
              en_func,
              scale_displ_sd,
              sd_parabolic_fit,
              interpolate_only,
              do_sd_corr,
              sd_corr_parabolic_fit,
              scale_displ_sd_corr,
              last_path,
              adapt_scale_disp,
              init_scale_displ_sd):
    """

    Everything here is in a.u., including xyz positions. They
    only get converged to Angstrom when used as input for
    an engrad calculation.


    g0_nxyz(np.array): nxyz of the initial geometry

    en_func(callable): a generic callable function
    that takes `nxyz` as input and returns the
    energy, the gradient, and the path where the calculation
    was performed. The gradient calculation is optional,
    and the calculation path is only for quantum chemistry,
    so that the orbitals from an energy calculation can be
    used to in a subsequent gradient calculation.


    """

    g1_nxyz, last_path, grad_0, g0_en = sd_before_correction(
        g0_nxyz=g0_nxyz,
        en_func=en_func,
        interpolate_only=interpolate_only,
        scale_displ_sd=scale_displ_sd,
        sd_parabolic_fit=sd_parabolic_fit,
        last_path=last_path)

    if not do_sd_corr:
        if adapt_scale_disp:
            scale_displ_sd = do_scale_adapt(
                g0_nxyz=g0_nxyz,
                gfinal_nxyz=g1_nxyz,
                scale_displ_sd=scale_displ_sd,
                init_scale_displ_sd=init_scale_displ_sd)

        return g1_nxyz, last_path, grad_0, g0_en, scale_displ_sd

    g1_en, grad_1, last_path = en_func(nxyz=g1_nxyz,
                                       grad=True,
                                       load_path=last_path)

    sdc_1_nxyz, sdc_1 = sd_corr(grad_0=grad_0,
                                grad_1=grad_1,
                                g0_nxyz=g0_nxyz,
                                g1_nxyz=g1_nxyz,
                                scale_displ_sd_corr=scale_displ_sd_corr)

    if not sd_corr_parabolic_fit:
        if adapt_scale_disp:
            scale_displ_sd = do_scale_adapt(
                g0_nxyz=g0_nxyz,
                gfinal_nxyz=sdc_1_nxyz,
                scale_displ_sd=scale_displ_sd,
                init_scale_displ_sd=init_scale_displ_sd)

        return sdc_1_nxyz, last_path, grad_0, g0_en, scale_displ_sd

    sdc_1_en, _, last_path = en_func(nxyz=sdc_1_nxyz,
                                     grad=False,
                                     load_path=last_path)
    sdc_2_nxyz = sd_corr_parabolic_step(sdc_1_nxyz=sdc_1_nxyz,
                                        sdc_1=sdc_1,
                                        g1_en=g1_en,
                                        sdc_1_en=sdc_1_en,
                                        g1_nxyz=g1_nxyz)
    sdc_2_en, _, _ = en_func(nxyz=sdc_2_nxyz,
                             grad=False,
                             load_path=last_path)

    gfinal_nxyz = sd_corr_parabolic(
        g1_nxyz=g1_nxyz,
        sdc_1_nxyz=sdc_1_nxyz,
        sdc_2_nxyz=sdc_2_nxyz,
        g1_en=g1_en,
        sdc_1_en=sdc_1_en,
        sdc_2_en=sdc_2_en,
        sdc_1=sdc_1,
        interpolate_only=interpolate_only)

    if adapt_scale_disp:
        scale_displ_sd = do_scale_adapt(
            g0_nxyz=g0_nxyz,
            gfinal_nxyz=gfinal_nxyz,
            scale_displ_sd=scale_displ_sd,
            init_scale_displ_sd=init_scale_displ_sd)

    return gfinal_nxyz, last_path, grad_0, g0_en, scale_displ_sd


def check_converged(grad,
                    tol_rms_g,
                    tol_max_g):

    grad = np.array(grad)
    rms_g = np.mean(grad ** 2) ** 0.5
    max_g = np.max(abs(grad))
    converged = (rms_g < tol_rms_g and
                 max_g < tol_max_g)

    return converged


def run_irc(g0_nxyz,
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
            adapt_scale_disp):

    trj = [g0_nxyz]
    ens = []

    init_scale_displ_sd = copy.deepcopy(scale_displ_sd)
    last_path = None
    converged = False

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

        g1_nxyz, last_path, grad_0, en_0, scale_displ_sd = out

        ens.append(float(en_0))
        converged = check_converged(grad=grad_0,
                                    tol_rms_g=tol_rms_g,
                                    tol_max_g=tol_max_g)

        if converged or num_steps == (max_steps - 1):
            if converged:
                print(f"IRC converged after {num_steps + 1} steps!")
            break

        trj.append(copy.deepcopy(g1_nxyz))
        g0_nxyz = g1_nxyz

    if not converged:
        print(f"IRC did not converge after {num_steps + 1} steps")

    return trj, ens


def convert_irc(irc):
    new_irc = []
    for nxyz in irc:
        nxyz_angs = copy.deepcopy(nxyz)
        nxyz_angs[:, 1:] *= const.BOHR_RADIUS
        new_irc.append(nxyz_angs)

    return new_irc


def back_convert_irc(irc):
    new_irc = []
    for nxyz in irc:
        nxyz_angs = copy.deepcopy(nxyz)
        nxyz_angs[:, 1:] /= const.BOHR_RADIUS
        new_irc.append(nxyz_angs)

    return new_irc


def run_all(ts_nxyz,
            eigvecs,
            freqs_cm,
            en_func,
            init_displ_de,
            max_iter,
            tol_max_g,
            tol_rms_g,
            scale_displ_sd_corr,
            sd_corr_parabolic_fit,
            do_sd_corr,
            interpolate_only,
            sd_parabolic_fit,
            scale_displ_sd,
            adapt_scale_disp,
            mode=-1):

    print("Generating initial displacements from TS... ")
    # input is ts_nxyz in Angstrom, output is in Bohr
    nxyz_fwd, nxyz_bckwd = init_displace(eigvecs=eigvecs,
                                         freqs_cm=freqs_cm,
                                         init_displ_de=init_displ_de,
                                         ts_nxyz=ts_nxyz,
                                         mode=mode)
    print("Generated displacements!")

    with open("init_displace.pickle", "wb") as f:
        pickle.dump([nxyz_fwd, nxyz_bckwd], f)

    ircs = []
    all_ens = []

    for i, g0_nxyz in enumerate([nxyz_fwd, nxyz_bckwd]):
        key = 'forward' if i == 0 else 'backward'
        print(f"Generating {key} IRC path")

        irc, ens = run_irc(g0_nxyz=g0_nxyz,
                           en_func=en_func,
                           scale_displ_sd=scale_displ_sd,
                           sd_parabolic_fit=sd_parabolic_fit,
                           interpolate_only=interpolate_only,
                           do_sd_corr=do_sd_corr,
                           sd_corr_parabolic_fit=sd_corr_parabolic_fit,
                           scale_displ_sd_corr=scale_displ_sd_corr,
                           tol_rms_g=tol_rms_g,
                           tol_max_g=tol_max_g,
                           max_steps=max_iter,
                           adapt_scale_disp=adapt_scale_disp)

        # convert final coordinates to Angstrom
        ircs.append(convert_irc(irc))
        # append energies
        all_ens.append(ens)

    # get the TS energy
    ts_en, _, _ = en_func(nxyz=ts_nxyz,
                          grad=False,
                          load_path=None,
                          convert=False)

    final_irc = [*list(reversed(ircs[0])),
                 ts_nxyz,
                 *ircs[1]]

    final_ens = [*list(reversed(all_ens[0])),
                 float(ts_en),
                 *all_ens[1]]

    return final_irc, final_ens


def make_working_dir(job_dir):
    folders = [int(i) for i in os.listdir(job_dir)
               if i.isdigit()]
    last_folder = sorted(folders)[-1]
    new_folder = os.path.join(job_dir,
                              str(last_folder + 1))
    os.makedirs(new_folder)

    return new_folder


def get_loader_fn(config_info,
                  config_path):

    loader_module_name = config_info["loader_module"]
    module_path = os.path.join(config_path, loader_module_name + ".py")
    spec = importlib.util.spec_from_file_location(loader_module_name,
                                                  module_path)
    loader_module = importlib.util.module_from_spec(spec)
    load_fn = loader_module.get_calc_list

    return load_fn


def parse(config_info,
          config_path,
          working_dir):

    load_calc_list = get_loader_fn(config_info=config_info,
                                   config_path=config_path)
    calcs = load_calc_list(job_dir=working_dir)
    assert len(calcs) == 1

    calc = calcs[0]
    props = calc.properties
    energy = props.totalenergy

    if 'forces' in props:
        grad = -np.array(props.forces)
    else:
        grad = None

    return energy, grad


def render_templates(working_dir,
                     config_info,
                     config_path,
                     jobspec):

    template_names = [config_info["job_template_filename"]
                      * config_info["extra_template_filenames"]]

    for template_name in template_names:
        template_path = os.path.join(config_path, template_name)
        write_path = os.path.join(working_dir, template_name)

        with open(template_path, 'r') as f:
            temp_text = f.read()

        template = Template(temp_text)
        inp = template.render(jobspec=jobspec)

        with open(write_path, 'w') as f_open:
            f_open.write(inp)


def get_config_info(config_name):
    direc = os.path.dirname(os.path.abspath(__file__))
    chemconfig_dir = os.path.join(direc, '../../../..',
                                  'chemconfigs')
    config_path = os.path.join(chemconfig_dir,
                               CONFIG_PATHS[config_name])
    config_json_path = os.path.join(config_path,
                                    'config.json')

    with open(config_json_path, 'r') as f:
        config_info = json.load(f)

    return config_path, config_info


def copy_from_old(last_path,
                  working_dir):

    for i in os.listdir(last_path):
        old_path = os.path.join(last_path, i)
        new_path = os.path.join(working_dir, i)

        if os.path.isfile(old_path):
            shutil.copy(old_path, new_path)
        elif os.path.isdir(old_path):
            shutil.copytree(old_path, new_path)


def nxyz_to_coords(nxyz):
    coords = [dict(element=PERIODICTABLE.GetElementSymbol(int(l[0])),
                   x=l[1],
                   y=l[2],
                   z=l[3]
                   ) for l in nxyz]
    return coords


def update_jobspec(jobspec,
                   nxyz,
                   grad_key,
                   needs_grad):

    coords = nxyz_to_coords(nxyz)
    jobspec['details'].update({"coords": coords,
                               grad_key: needs_grad})


def run_calc(config_name,
             jobspec,
             job_dir,
             last_path,
             needs_grad,
             grad_key,
             nxyz):
    """
    To-do: Add option to provide path to seed the SCF calculation
    """

    config_path, config_info = get_config_info(config_name)
    working_dir = make_working_dir(job_dir)

    if last_path is not None:
        copy_from_old(last_path=last_path,
                      working_dir=working_dir)

    update_jobspec(jobspec=jobspec,
                   nxyz=nxyz,
                   grad_key=grad_key,
                   needs_grad=needs_grad)

    render_templates(working_dir=working_dir,
                     config_info=config_info,
                     config_path=config_path,
                     jobspec=jobspec)

    os.chdir(working_dir)
    bash_command("bash job.sh")
    os.chdir(job_dir)

    energy, grad = parse(config_info=config_info,
                         config_path=config_path,
                         working_dir=working_dir)

    return energy, grad, working_dir


def run_model(nxyz,
              model,
              cutoff,
              neural_en_key,
              device,
              model_kwargs):

    props = {"nxyz": [nxyz]}
    dset = Dataset(props)
    dset.generate_neighbor_list(cutoff=cutoff,
                                undirected=False)

    loader = DataLoader(dset, collate_fn=collate_dicts)
    batch = batch_to(next(iter(loader)), device)

    kwargs = model_kwargs if model_kwargs is not None else {}
    results = batch_detach(model(batch, **kwargs))

    conv_results = {}
    for key, val in results.items():

        if not any([key == neural_en_key,
                    key == neural_en_key + "_grad"]):
            continue

        conv = copy.deepcopy(const.KCAL_TO_AU['energy'])
        use_key = 'energy'

        if key.endswith('_grad'):
            use_key = 'grad'
            conv *= const.KCAL_TO_AU['_grad']

        conv_val = conv * val.numpy()
        conv_results[use_key] = conv_val

    energy = conv_results['energy']
    grad = conv_results['grad']

    return energy, grad


def make_en_func(calc_type,
                 neural_en_key=None,
                 cutoff=None,
                 device=None,
                 model_path=None,
                 config_name=None,
                 jobspec=None,
                 job_dir=None,
                 grad_key=None,
                 model_kwargs=None,
                 model=None):

    if calc_type == 'dft':
        def func(nxyz,
                 grad=False,
                 load_path=None,
                 convert=True):

            if convert:
                # convert nxyz to Angstrom
                nxyz_angs = copy.deepcopy(nxyz)
                nxyz_angs[:, 1:] *= const.BOHR_RADIUS
            else:
                nxyz_angs = nxyz

            out = run_calc(config_name=config_name,
                           jobspec=jobspec,
                           job_dir=job_dir,
                           last_path=load_path,
                           needs_grad=grad,
                           grad_key=grad_key,
                           nxyz=nxyz_angs)

            return out

    elif calc_type == 'neural':
        if model is None:
            model = load_model(model_path)
            model = model.to(device)

        def func(nxyz,
                 grad=True,
                 load_path=None,
                 convert=True):

            if convert:
                # convert nxyz to Angstrom
                nxyz_angs = copy.deepcopy(nxyz)
                nxyz_angs[:, 1:] *= const.BOHR_RADIUS
            else:
                nxyz_angs = nxyz

            energy, grad = run_model(nxyz=nxyz_angs,
                                     model=model,
                                     cutoff=cutoff,
                                     neural_en_key=neural_en_key,
                                     device=device,
                                     model_kwargs=model_kwargs)
            out = energy, grad, None

            return out

    else:

        raise NotImplementedError

    return func


def save(job_dir,
         irc,
         energies):

    trj = [Atoms(numbers=nxyz[:, 0],
                 positions=nxyz[:, 1:])
           for nxyz in irc]
    dic = {"nxyz_list": irc,
           "atoms": trj,
           "energies": energies}

    save_path = os.path.join(job_dir, 'irc.pickle')

    with open(save_path, 'wb') as f:
        pickle.dump(dic, f)

    print(f"IRC saved to {save_path}")


def main(ts_nxyz,
         eigvecs,
         freqs_cm,
         init_displ_de,
         max_iter,
         tol_max_g,
         tol_rms_g,
         scale_displ_sd_corr,
         sd_corr_parabolic_fit,
         do_sd_corr,
         interpolate_only,
         sd_parabolic_fit,
         scale_displ_sd,
         adapt_scale_disp,
         calc_type,
         jobspec,
         job_dir,
         model_path=None,
         cutoff=None,
         device=None,
         neural_en_key=None,
         config_name=None,
         grad_key=None,
         mode=-1,
         model_kwargs=None):

    # grad_key here is the key that you supply in
    # `details` to say whether or not to take a grad.
    # For us we'll have to change that to grad_roots
    # and just set it to blank, because there's no other
    # way to just get the energy

    try:
        en_func = make_en_func(calc_type=calc_type,
                               neural_en_key=neural_en_key,
                               cutoff=cutoff,
                               device=device,
                               model_path=model_path,
                               config_name=config_name,
                               jobspec=jobspec,
                               job_dir=job_dir,
                               grad_key=grad_key,
                               model_kwargs=model_kwargs)

        irc, energies = run_all(ts_nxyz=ts_nxyz,
                                eigvecs=eigvecs,
                                freqs_cm=freqs_cm,
                                en_func=en_func,
                                init_displ_de=init_displ_de,
                                max_iter=max_iter,
                                tol_max_g=tol_max_g,
                                tol_rms_g=tol_rms_g,
                                scale_displ_sd_corr=scale_displ_sd_corr,
                                sd_corr_parabolic_fit=sd_corr_parabolic_fit,
                                do_sd_corr=do_sd_corr,
                                interpolate_only=interpolate_only,
                                sd_parabolic_fit=sd_parabolic_fit,
                                scale_displ_sd=scale_displ_sd,
                                adapt_scale_disp=adapt_scale_disp,
                                mode=mode)

    except Exception as e:
        print(e)
        import pdb
        pdb.post_mortem()

    save(job_dir=job_dir,
         irc=irc,
         energies=energies)


def load_info(info_path):
    with open(info_path, 'r') as f:
        info = json.load(f)
    info = {**info, **info['details']}
    info.pop('details')

    return info


def coords_to_nxyz(coords):
    nxyz = []
    for dic in coords:
        n = PERIODICTABLE.GetAtomicNumber(dic['element'])
        xyz = [dic['x'], dic['y'], dic['z']]
        nxyz.append([n, *xyz])

    nxyz = np.array(nxyz)

    return nxyz


def get_model_path(info):

    nnid = info['nnid']
    weightpath = info['weightpath']
    if not os.path.isdir(weightpath):
        weightpath = info.get('mounted_weightpath', '')

    if not os.path.isdir(weightpath):
        raise Exception(f"{weightpath} is not a directory")

    model_path = os.path.join(weightpath, str(nnid))

    return model_path


def get_modes(job_dir,
              ts_nxyz,
              model_path,
              neural_en_key,
              device,
              cutoff,
              model_kwargs):

    # get the calculator
    nff = NeuralFF.from_file(model_path,
                             en_key=neural_en_key,
                             device=device,
                             model_kwargs=model_kwargs)

    directed = not (type(nff.model) in UNDIRECTED)

    # make atoms batch and give it the calculator

    atoms_batch = AtomsBatch(numbers=ts_nxyz[:, 0],
                             positions=ts_nxyz[:, 1:],
                             directed=directed,
                             device=device,
                             cutoff=cutoff)

    atoms_batch.set_calculator(nff)

    # removed any old frequency directories that
    # could mess up the calculation

    vib_dir = os.path.join(job_dir, 'vib')
    if os.path.isdir(vib_dir):
        shutil.rmtree(vib_dir)

    # get the modes and eigenvalues
    results = hessian_and_modes(atoms_batch)
    eigvecs = results['modes']
    freqs_cm = np.array(results['vibfreqs'])

    return eigvecs, freqs_cm


def neural_from_file(job_dir,
                     info_path):

    # load some info needed for the main function

    info = load_info(info_path)
    ts_nxyz = coords_to_nxyz(info['coords'])
    model_path = get_model_path(info)

    # do a Hessian calculation

    device = info["device"]
    cutoff = info["cutoff"]
    neural_en_key = info["en_key"]
    model_kwargs = info.get("model_kwargs")

    eigvecs, freqs_cm = get_modes(job_dir=job_dir,
                                  ts_nxyz=ts_nxyz,
                                  model_path=model_path,
                                  neural_en_key=neural_en_key,
                                  device=device,
                                  cutoff=cutoff,
                                  model_kwargs=model_kwargs)
    # Run IRC

    main(ts_nxyz=ts_nxyz,
         eigvecs=eigvecs,
         freqs_cm=freqs_cm,
         init_displ_de=info["init_displ_de"],
         max_iter=info["max_iter"],
         tol_max_g=info["tol_max_g"],
         tol_rms_g=info["tol_rms_g"],
         scale_displ_sd_corr=info["scale_displ_sd_corr"],
         sd_corr_parabolic_fit=info["sd_corr_parabolic_fit"],
         do_sd_corr=info["do_sd_corr"],
         interpolate_only=info["interpolate_only"],
         sd_parabolic_fit=info["sd_parabolic_fit"],
         scale_displ_sd=info["scale_displ_sd"],
         adapt_scale_disp=info["adapt_scale_disp"],
         calc_type='neural',
         jobspec=None,
         job_dir=job_dir,
         model_path=model_path,
         cutoff=cutoff,
         device=device,
         neural_en_key=neural_en_key,
         mode=-1)

    print(COMPLETION_MESSAGE)


def run_from_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--info_file',
                        type=str,
                        default='job_info.json',
                        help=("Name of the job info file"))

    args = parser.parse_args()

    neural_from_file(job_dir=".",
                     info_path=args.info_file)


if __name__ == "__main__":
    run_from_command_line()

"""
Similar to neuraloptimizer, but specifically designed to make a conformer
ensemble like CREST, with enhanced sampling and repeated MD when lower-energy
conformers are found. Run `python neural_confgen.py --help` for a list of all the
required and optional arguments. See `djangochem/neuralnet/json_files/neural_confgen.json`
and/or `chemconfigs/nnpotential/neural_confgen/default_details.json` for default
arguments.
"""

from rdkit.Chem import BondType
from rdkit import Chem
import copy
import numpy as np
import os
import shutil
import random
import string as string_module
import pickle
import time
import json

from ase.io.trajectory import TrajectoryWriter, Trajectory
from ase.constraints import FixInternals, FixBondLengths

from nff.utils import constants as const
from nff.utils.misc import bash_command
from nff.utils.misc import parse_args_from_json as parse_args
from barriers.utils.neuraloptimizer import (md_to_conf,
                                            add_all_constraints,
                                            params_to_atoms,
                                            reset_h_mass,
                                            get_trj_file,
                                            get_model,
                                            OPT_FILENAME,
                                            EN_PATH)
from barriers.utils.ase_neb import load_params

from barriers.utils.neuraloptimizer import confs_to_opt as base_opt

FINAL_OPT_FILENAME = "final_opt.traj"
ARGS_PATH = 'json_files/args.json'
JSON_KEYS = ['model_kwargs',
             'constraints',
             'enhanced_sampling',
             'sample_rate_fs',
             'crest_dedupe']


BOS = {BondType.SINGLE: 1.0,
       BondType.DOUBLE: 2.0,
       BondType.TRIPLE: 3.0,
       BondType.AROMATIC: 1.5,
       BondType.UNSPECIFIED: 0.0}


def non_terminal_bonds(mol):

    bonds = mol.GetBonds()
    non_terminal = []

    for bond in bonds:
        start_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()

        min_num_bonds = min([i.GetDegree() for i in [start_atom, end_atom]])
        if min_num_bonds <= 1:
            continue
        non_terminal.append(bond)

    return non_terminal


def get_flex(mol):

    bonds = non_terminal_bonds(mol)
    av2 = 0
    for bond in bonds:
        hybf = 1

        start = bond.GetBeginAtom()
        end = bond.GetEndAtom()

        atoms = [start, end]
        cns = [atom.GetDegree() for atom in atoms]
        if start.GetAtomicNum() == 6 and cns[0] <= 3.3:
            hybf = 1.2
        if end.GetAtomicNum() == 6 and cns[1] <= 3.3:
            hybf = 1.2

        n_neigh_a = start.GetDegree()
        n_neigh_b = end.GetDegree()

        bo = BOS[bond.GetBondType()]
        doublef = 1.0 - np.exp(-4.0 * (bo - 2.0) ** 6)
        branch = 2.0 / np.sqrt(n_neigh_a * n_neigh_b)

        ring_sizes = []
        for atom in atoms:
            in_ring = atom.IsInRing()
            if in_ring:
                for size in range(100):
                    in_this_size = atom.IsInRingSize(size)
                    if in_this_size:
                        break
                ring_sizes.append(size)
            else:
                ring_sizes.append(0)

        min_ring_size = min(ring_sizes)
        if min_ring_size == 0:
            ringf = 1
        else:
            ringf = 0.5 * (1.0 - np.exp(-0.06 * min_ring_size))

        val = branch * ringf * doublef * hybf
        av2 += val ** 2

    all_bonds = mol.GetBonds()
    av2 = np.sqrt(av2 / len(all_bonds))

    return av2


def get_t_mtd(mol,
              num_fixed):

    flex = get_flex(mol)
    rednat = len(mol.GetAtoms()) - num_fixed
    av1 = flex * max([1, rednat - 8])
    t_mtd = np.round(3.0 * np.exp(0.10 * av1))

    # We set t_mtd to be between 15 and 200 ps
    t_mtd = max([15.0, t_mtd])
    t_mtd = min([t_mtd, 200.0])

    return t_mtd


def clean_atoms(atoms):
    """
    Remove any constraints and weird hydrogen masses used for mTD
    """

    keep_constraints = [i for i in atoms.constraints if not
                        isinstance(i, FixInternals) and not
                        isinstance(i, FixBondLengths)]
    atoms.constraints = keep_constraints
    reset_h_mass(atoms)

    return atoms


def get_num_starting_poses(params,
                           md_type=None):

    sample_rate = params["sample_rate_fs"]
    # sample rate can be a dictionary or a number
    if isinstance(sample_rate, dict):
        sample_rate = sample_rate[md_type]

    num_starting_poses = int(np.floor(params["steps"] * params["time_step"] /
                                      sample_rate))

    return num_starting_poses


def update_params(params):
    """
    Get parameters that needed to be specified in `neuraloptimizer` that are now
    fixed in `neural_confgen`. Or params that have new names so they can be used
    with the old functions
    """

    new_params = {"do_md": True,
                  "do_save": True,
                  "assert_converged": False,
                  "max_rounds": params.get("max_restart_opt")}

    params.update(new_params)


def run_md_before_gamd(atoms,
                       params,
                       model):

    gamd_params = params["enhanced_sampling"]["params"]
    init_md_steps = gamd_params["init_md_steps"]

    init_params = copy.deepcopy(params)

    init_params.update({"steps": init_md_steps})
    num_starting_poses = get_num_starting_poses(init_params,
                                                md_type="MD")
    init_params.update({"num_starting_poses": num_starting_poses})
    init_params.pop("enhanced_sampling")

    if atoms is not None:
        init_params.update({"coords": atoms_to_coords(atoms),
                            "nxyz": atoms_to_nxyz(atoms)})

    atoms = params_to_atoms(init_params,
                            model=model)
    confs = md_to_conf(atoms=atoms,
                       params=init_params,
                       model=model)

    return confs


def atoms_to_nxyz(atoms):
    nums = atoms.get_atomic_numbers()
    pos = atoms.get_positions()
    nxyz = np.concatenate([nums.reshape(-1, 1),
                           pos], axis=-1).tolist()

    return nxyz


def atoms_to_coords(atoms):
    elements = atoms.get_chemical_symbols()
    pos = atoms.get_positions()

    coords = []
    for element, xyz in zip(elements, pos):
        coord = {"element": str(element),
                 "x": float(xyz[0]),
                 "y": float(xyz[1]),
                 "z": float(xyz[2])}
        coords.append(coord)
    return coords


def get_params_for_gamd(params,
                        model):
    """
    Caution! Make sure "atoms.traj" is always the latest run, so we're not re-using
    the first run over and over
    """

    trj_file = get_trj_file(params)
    trj = Trajectory(trj_file)

    ens = np.array([atoms.get_potential_energy().item()
                    for atoms in trj]).reshape(-1)
    # ignore the first frame because it's close to converged and so much lower
    # in energy than any of the other frames from MD
    min_en = min(ens[1:])
    max_en = max(ens[1:])

    gamd_params = copy.deepcopy(params["enhanced_sampling"]["params"])
    gamd_params.update({"V_min": min_en,
                        "V_max": max_en})
    new_params = copy.deepcopy(params)
    new_params["enhanced_sampling"]["params"] = gamd_params
    new_params.update({'coords': atoms_to_coords(trj[-1]),
                       "nxyz": atoms_to_nxyz(trj[-1]),
                       'num_starting_poses': get_num_starting_poses(new_params,
                                                                    md_type="GAMD")})

    atoms = params_to_atoms(new_params,
                            model=model)

    return new_params, atoms


def move_general_file(file,
                      job_dir=None):

    if job_dir is None:
        job_dir = os.getcwd()

    path = os.path.join(job_dir, file)
    if not os.path.isfile(path):
        return

    split = path.split(".")
    start = ".".join(split[:-1])
    end = split[-1]

    i = 0
    while True:
        new_name = "%s_%d.%s" % (start, i, end)
        if not os.path.isfile(new_name):
            break
        i += 1

    shutil.move(path, new_name)

    return new_name


def move_trj_file(params,
                  job_dir=None):
    trj_file = get_trj_file(params)
    move_general_file(trj_file,
                      job_dir=job_dir)


def move_opt_file(job_dir=None):
    move_general_file(OPT_FILENAME,
                      job_dir=job_dir)


def run_gamd(atoms,
             params,
             model):

    # move the old trajectory file if it's lying around
    move_trj_file(params=params)

    # get conformers from the initial MD run
    print("Running normal MD to get parameters for GAMD...")
    confs = run_md_before_gamd(atoms=atoms,
                               params=params,
                               model=model)
    print("Normal MD complete!")

    # use that run to get parameters for subsequent GAMD
    new_params, new_atoms = get_params_for_gamd(params=params,
                                                model=model)
    # move the MD file
    move_trj_file(params=params)

    # run GAMD
    print("Running GAMD...")
    new_confs = md_to_conf(atoms=new_atoms,
                           params=new_params,
                           model=model)
    print("GAMD complete!")

    # move the old trajectory file
    move_trj_file(params=params)

    all_confs = confs + new_confs

    return all_confs


def run_single_md(atoms,
                  params,
                  model):
    """
    Function for running any type of MD that is one part (i.e. not two stages like
    GAMD)
    """

    new_params = copy.deepcopy(params)
    num_starting_poses = get_num_starting_poses(new_params,
                                                md_type="MD")
    new_params.update({"num_starting_poses": num_starting_poses})
    if atoms is not None:
        new_params.update({"coords": atoms_to_coords(atoms),
                           "nxyz": atoms_to_nxyz(atoms)})
    atoms = params_to_atoms(new_params,
                            model=model)

    move_trj_file(params=params)
    print("Running dynamics...")
    confs = md_to_conf(atoms=atoms,
                       params=new_params,
                       model=model)

    # remove anything you used in the dynamics that you don't want during opt,
    # like increased hydrogen mass and fixed bond lengths
    confs = [clean_atoms(i) for i in confs]

    print("Dynamics complete!")
    move_trj_file(params=params)

    return confs


def confs_to_opt(params,
                 best_confs,
                 model):

    # if the optimization constraints are the same as the MD constraints,
    # just use `base_opt` (i.e. `confs_to_opt` in `neuraloptimizer`)

    opt_constraints = params.get("opt_constraints")
    if opt_constraints is None:
        return base_opt(params=params,
                        best_confs=best_confs,
                        model=model)

    # if the constraints are different, then make `new_params`, remove any
    # of the old constraint keys, and update with the opt constraint key

    new_params = copy.deepcopy(params)
    for key in ["constraints", "fixed_atoms"]:
        if key in new_params:
            new_params.pop(key)
    new_params.update(opt_constraints)

    # add the opt constraints from `new_params`
    for i, atoms in enumerate(best_confs):
        atoms.constraints = []
        add_all_constraints(atoms=atoms,
                            params=new_params)
        best_confs[i] = atoms

    # optimize
    sorted_atoms, sorted_ens = base_opt(params=params,
                                        best_confs=best_confs,
                                        model=model)

    # remove the opt constraints and add back the MD constraints from `params`

    for i, atoms in enumerate(sorted_atoms):
        atoms.constraints = []
        add_all_constraints(atoms=atoms,
                            params=params)
        sorted_atoms[i] = atoms

    return sorted_atoms, sorted_ens


def coarse_opt(frames,
               params,
               model):

    coarse_params = copy.deepcopy(params)
    coarse_params.update({"fmax": params["fmax_coarse"]})

    print("Optimizing %d conformers..." % (len(frames)))
    sorted_atoms, sorted_ens = confs_to_opt(params=coarse_params,
                                            best_confs=frames,
                                            model=model)
    sorted_atoms, _ = trim(sorted_atoms, params["window_coarse"])

    print("Completed crude optimization")
    move_opt_file()

    return sorted_atoms


def fine_opt(coarse_confs,
             params,
             model):

    fine_params = copy.deepcopy(params)
    fine_params.update({"fmax": params["fmax_tight"]})

    print("Optimizing %d conformers..." % (len(coarse_confs)))
    sorted_atoms, sorted_ens = confs_to_opt(params=fine_params,
                                            best_confs=coarse_confs,
                                            model=model)
    sorted_atoms, final_ens = trim(sorted_atoms, params["window_tight"])

    print("Completed tight optimization")
    move_opt_file()

    return sorted_atoms, final_ens


def get_is_done(old_ens,
                new_ens,
                params):

    e_thresh = params["lower_e_tol"] / const.EV_TO_KCAL_MOL
    new_min_en = np.min(new_ens)

    old_min_en = np.min(old_ens)
    is_done = (new_min_en > (old_min_en - e_thresh))
    print("Old lowest energy: %.4f eV" % old_min_en)
    print("New lowest energy: %.4f eV" % new_min_en)
    if is_done:
        print(("New energy is no more than %.2f kcal/mol "
               "below the old energy. Breaking now.") % (params["lower_e_tol"]))
    else:
        delta_e = (old_min_en - new_min_en) * const.EV_TO_KCAL_MOL
        print(("New energy is %.2f kcal/mol lower than the old energy, "
               "which is below the user threshold of %.2f kcal/mol. Continuing confgen."
               ) % (delta_e, params["lower_e_tol"]))

    return is_done


def get_best_conf(all_confs,
                  all_ens):

    argsort = np.argsort(all_ens)
    best_idx = argsort[0]
    best_conf = copy.deepcopy(all_confs[best_idx])

    return best_conf


def make_rand_string(num_letters=10):
    string = "".join(random.sample(string_module.ascii_letters,
                                   num_letters))
    return string


def base_read_unique(job_dir):
    path = os.path.join(job_dir, "enso.tags")
    if not os.path.isfile(path):
        return

    with open(path, 'r') as f:
        lines = f.readlines()
    unique_idx = []
    for line in lines:
        split = line.strip().split()
        if not split:
            continue

        idx = split[-1].split("!CONF")[-1]
        # means something went wrong
        if not idx.isdigit():
            return

        unique_idx.append(int(idx) - 1)

    return unique_idx


def read_unique(job_dir):
    try:
        return base_read_unique(job_dir)
    except Exception as e:
        print(e)
        print("Keeping all conformers")
        return


def set_xtb_env(xtb_dir):

    commands = ["XTBHOME=%s" % xtb_dir,
                "export XTBPATH=${XTBHOME}/share/xtb:${XTBHOME}:${HOME}",
                "MANPATH=${MANPATH}:${XTBHOME}/share/man",
                "PATH=${PATH}:${XTBHOME}/bin",
                "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${XTBHOME}/lib64",
                "PYTHONPATH=${PYTHONPATH}:${XTBHOME}/python",
                "export PATH XTBPATH MANPATH LD_LIBRARY_PATH PYTHONPATH"]

    for command in commands:
        p = bash_command(command)
        p.wait()


def make_xyz_text(positions,
                  symbols,
                  comment):

    num_atoms = len(positions)

    lines = [str(num_atoms), comment]

    for symbol, this_pos in zip(symbols, positions):
        line = "%s %.8f %.8f %.8f " % (symbol,
                                       this_pos[0],
                                       this_pos[1],
                                       this_pos[2])
        lines.append(line)

    text = "\n".join(lines)
    return text


def write_confs_xyz(confs, path):

    energies = np.array([float(atoms.get_potential_energy()) * const.EV_TO_AU
                         for atoms in confs])

    # Crazy weird bug where an energy of 0 means the conformer gets removed
    # In that case, add a constant shift, which shouldn't affect any of the results

    if (abs(energies) < 1e-3).any():
        energies += (-energies.min() + 0.5)

    text = ""
    for i, atoms in enumerate(confs):
        # energy in Hartree
        energy = energies[i]

        comment = "%.8f !CONF%d" % (energy, i + 1)

        this_text = make_xyz_text(positions=atoms.get_positions(),
                                  symbols=atoms.get_chemical_symbols(),
                                  comment=comment)

        if i != 0:
            text += "\n"
        text += this_text

    with open(path, 'w') as f:
        f.write(text)


def run_cre_check(confs,
                  ethr,
                  rthr,
                  bthr,
                  ewin):

    crest_path = os.path.join(os.environ["CONDA_PREFIX"], 'bin/crest')
    base_name = make_rand_string()
    job_dir = os.path.join("/tmp", base_name)

    cwd = os.getcwd()
    os.makedirs(job_dir)
    os.chdir(job_dir)

    confs_path = os.path.join(job_dir, "confs.xyz")
    conf_0_path = os.path.join(job_dir, "conf_0.xyz")

    write_confs_xyz(confs, path=confs_path)
    write_confs_xyz(confs[:1], path=conf_0_path)

    # set_xtb_env(xtb_dir)
    command = ("%s %s -cregen %s -ethr %.6f -rthr %.6f -bthr %.6f -ewin %.6f -enso "
               "> cregen.out" % (crest_path, conf_0_path, confs_path, ethr, rthr,
                                 bthr, ewin))

    p = bash_command(command)
    p.wait()

    unique_idx = read_unique(job_dir=job_dir)
    if unique_idx is None:
        return confs

    num_removed = len(confs) - len(unique_idx)

    plural = 's' if num_removed > 1 else ''
    print("Removed %d duplicate conformer%s with cregen" % (num_removed,
                                                            plural))

    confs = [confs[i] for i in unique_idx]

    shutil.rmtree(job_dir)
    os.chdir(cwd)

    return confs


def save_final_confs(confs):
    trj_writer = TrajectoryWriter(filename=FINAL_OPT_FILENAME)
    for atoms in confs:
        trj_writer.write(atoms)
    trj_writer.close()

    ens = np.array([i.get_potential_energy().item()
                    for i in confs]).reshape(-1).tolist()
    with open(EN_PATH, 'wb') as f:
        pickle.dump(ens, f)

    print("%d confs saved" % (len(confs)))


def get_md_function(params):
    sampling_method = params["enhanced_sampling"]["method"]
    if sampling_method.upper() == "GAMD":
        md_function = run_gamd
    elif "metadynamics" in sampling_method.lower():
        md_function = run_single_md
    else:
        raise NotImplementedError

    return md_function


def parse_path(path):
    direc = os.path.dirname(os.path.realpath(__file__))
    ext_dir = os.path.join(direc, "../../ext_programs")
    path = (path.replace("$HOME", os.environ["HOME"])
            .replace("ext_programs", ext_dir))
    return path


def initial_opt(params, model):

    print("Optimizing input conformer with tight thresholds...")

    these_params = copy.deepcopy(params)
    these_params.pop("enhanced_sampling")
    these_params["fmax"] = these_params["fmax_tight"]
    atoms = params_to_atoms(these_params,
                            model=model)
    # in case bond lengths are constrained and H masses are set to 2 for mTD
    atoms = clean_atoms(atoms)

    best_atoms, sorted_ens = confs_to_opt(params=these_params,
                                          best_confs=[atoms],
                                          model=model)

    print("Initial optimization complete!")

    return best_atoms, sorted_ens


def time_to_msg(seconds):
    total_hours = np.floor(seconds / 3600)

    remaining = seconds - total_hours * 3600
    total_minutes = np.floor(remaining / 60)

    remaining = seconds - total_hours * 3600 - total_minutes * 60
    total_seconds = remaining

    msg = ("Finished in %d hours, %d minutes, and %d seconds" % (
        total_hours, total_minutes, total_seconds))

    return msg


def report_time(start,
                end):
    seconds = end - start
    msg = time_to_msg(seconds)
    print(msg)


def dedupe(confs, params):
    dedupe_params = params["crest_dedupe"]
    if not dedupe_params["on"]:
        print("De-duplication turned off by user request.")
        return confs

    crest_params = dedupe_params["params"]
    confs = run_cre_check(confs=confs,
                          ethr=crest_params["ethr"],
                          rthr=crest_params["rthr"],
                          bthr=crest_params["bthr"],
                          ewin=crest_params["ewin"])
    ens = [conf.get_potential_energy().item() for conf in confs]

    return confs, ens


def trim(confs, window):
    ens = np.array([conf.get_potential_energy().item()
                    for conf in confs]).reshape(-1)
    rel_ens = (ens - np.min(ens)) * const.EV_TO_KCAL_MOL
    valid_idx = (rel_ens <= window).nonzero()[0]

    confs = [confs[i] for i in valid_idx]
    ens = [ens[i] for i in valid_idx]

    vowel_ints = [8, 11, 18]
    suffix = "n" if any([str(window).startswith(str(i))
                         for i in vowel_ints]) else ""
    print("%d conformers remain within a%s %.2f kcal/mol window" % (
        len(confs), suffix, window))

    return confs, ens


def summarize_final(confs,
                    params):

    suffix = "s" if len(confs) > 1 else ""
    print("%d conformer%s in the final CRE within a %.2f kcal/mol window" % (
        len(confs), suffix, params["window_vtight"]))

    ens = np.array([conf.get_potential_energy().item()
                    for conf in confs]).reshape(-1)
    rel_ens = (ens - np.min(ens)) * const.EV_TO_KCAL_MOL
    en_string = "\n".join(["%d:    %.4f" % ((i + 1), float(en)) for i, en in
                           enumerate(rel_ens)])
    print("Energies (kcal/mol):\n%s" % en_string)


def final_opt(all_confs,
              params,
              model):

    all_confs, _ = trim(confs=all_confs, window=params["window_tight"])

    final_params = copy.deepcopy(params)
    final_params.update({"fmax": params["fmax_vtight"]})

    print("Optimizing %d conformers..." % (len(all_confs)))
    all_confs, _ = confs_to_opt(params=final_params,
                                best_confs=all_confs,
                                model=model)
    all_confs, _ = trim(confs=all_confs, window=params["window_vtight"])

    print("Completed final ultra-tight optimization")
    move_opt_file()

    return all_confs


def make_confs(params,
               model=None,
               job_dir=None):

    start = time.time()

    if model is None:
        model = get_model(params)
    if job_dir is not None:
        os.chdir(job_dir)

    md_function = get_md_function(params)
    all_confs, all_ens = initial_opt(params,
                                     model=model)
    atoms = copy.deepcopy(all_confs[0])

    is_done = False
    for num_md_runs in range(params["max_md_runs"]):

        print("Running MD...")

        sampled_frames = md_function(atoms=atoms,
                                     params=params,
                                     model=model)

        print("MD complete!")
        print("Optimizing with loose thresholds...")

        # don't use first frame - that was already tightly optimized
        # from the previous round, whereas the others are only loosely optimized,
        # so we don't want that to throw off the thresholding

        coarse_confs = coarse_opt(frames=sampled_frames[1:],
                                  params=params,
                                  model=model)

        print("Removing duplicates among loosely-optimized conformers...")
        coarse_confs, _ = dedupe(confs=coarse_confs,
                                 params=params)

        print("Optimizing with tight thresholds...")
        fine_confs, fine_ens = fine_opt(coarse_confs=coarse_confs,
                                        params=params,
                                        model=model)
        print("Removing duplicates among tightly-optimized conformers...")
        fine_confs, fine_ens = dedupe(confs=fine_confs,
                                      params=params)

        is_done = get_is_done(old_ens=all_ens,
                              new_ens=fine_ens,
                              params=params)

        if num_md_runs < params["min_md_runs"]:
            is_done = False

        all_confs += fine_confs
        all_ens += fine_ens

        print("Removing duplicates among all conformers...")
        all_confs, all_ens = dedupe(confs=all_confs,
                                    params=params)

        print("%d conformers are in the CRE" % (len(all_confs)))

        if is_done:
            break

        atoms = get_best_conf(all_confs=all_confs,
                              all_ens=all_ens)

        # pick up from where the last trajectory left off, not from the lowest
        # energy geometry. We don't want to go back into the same region of space
        # that we already sampled

        # atoms = sampled_frames[-1]

    if not is_done:
        print("Reached the maximum of %d MD runs" % (params["max_md_runs"]))

    print("Optimizing final CRE with ultra-tight thresholds...")
    all_confs = final_opt(all_confs=all_confs,
                          params=params,
                          model=model)

    print("Removing duplicates among all conformers in the final CRE...")
    all_confs, _ = dedupe(confs=all_confs,
                          params=params)

    summarize_final(confs=all_confs,
                    params=params)

    print("Saving CRE...")
    save_final_confs(confs=all_confs)

    print("Complete!")

    end = time.time()
    report_time(start=start,
                end=end)


def get_num_fixed(params):
    fixed_idx = []

    dics = [params.get("fixed_atoms", {})]
    dics += [params.get("hookean", {}).get(key, {})
             for key in ['bonds', 'angles', 'dihedrals']]
    for dic in dics:
        idx = dic.get("idx")
        if idx is not None:
            fixed_idx += idx

    fixed_idx = list(set(fixed_idx))
    num_fixed = len(fixed_idx)

    return num_fixed


def set_mtd_time(info_file,
                 params):

    if not params.get("infer_time_from_flex", True):
        return

    smiles = params['smiles']

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    num_fixed = get_num_fixed(params)
    mtd_time = get_t_mtd(mol=mol,
                         num_fixed=num_fixed)
    params['mtd_time'] = mtd_time
    params["steps"] = int(np.ceil(params["mtd_time"] * 1000 /
                                  params["time_step"]))

    with open(info_file, 'w') as f:
        json.dump(params, f, indent=4)


def run(params):
    update_params(params)
    make_confs(params=params)


def parse():
    direc = os.path.dirname(os.path.abspath(__file__))
    args = parse_args(arg_path=ARGS_PATH,
                      direc=direc)

    params = args.__dict__
    info_file = args.info_file
    if info_file is not None:
        if not os.path.isfile(info_file):
            raise Exception(("Can't find the requested info "
                             "file %s" % info_file))
        these_params = load_params(file=args.info_file)
        set_mtd_time(info_file=info_file,
                     params=these_params)
        params.update(these_params)

    # if anything is supposed to be a dictionary and was given as a string
    # on the command line, convert now

    for key in JSON_KEYS:
        val = params.get(key)
        try:
            params[key] = json.loads(val)
        except TypeError:
            continue

    return params


def main():
    params = parse()
    run(params=params)


if __name__ == "__main__":
    main()

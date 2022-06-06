import numpy as np
import os
import json
from torch.utils.data import DataLoader
from torch.nn.modules.container import ModuleDict
import copy
import pickle
from rdkit import Chem
import argparse

from ase import optimize
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import TrajectoryWriter
from ase.io.trajectory import Trajectory as AseTrajectory
from ase.constraints import (FixAtoms, FixInternals, FixBondLengths)
from ase.constraints import Hookean
from ase.io.jsonio import read_json, write_json

from nff.io.ase import (NeuralFF,
                        AtomsBatch,
                        UNDIRECTED,
                        NeuralMetadynamics,
                        BatchNeuralMetadynamics,
                        NeuralGAMD)
from nff.train import load_model
from nff.data import collate_dicts, Dataset
from nff.md import nve, nvt
from nff.utils.constants import EV_TO_AU, BOHR_RADIUS, EV_TO_KCAL_MOL


from nff.io.ase_utils import (ConstrainDihedrals, ConstrainAngles,
                              ConstrainBonds, BatchedBFGS, BatchedLBFGS)

from barriers.irc.neural_irc import init_displace
from barriers.utils.vib import hessian_and_modes

PERIODICTABLE = Chem.GetPeriodicTable()

RESTART_FILE = "restart.json"
OPT_KEYS = ["steps", "fmax"]
MAX_ROUNDS = 5
NUM_CONFS = 15
OPT_FILENAME = "opt.traj"
EN_PATH = "energies.pickle"
DEFAULT_INFO_FILE = "job_info.json"

INTEGRATOR_DIC = {"velocityverlet": VelocityVerlet}
METHOD_DIC = {
    "nosehoover": nvt.NoseHoover,
    "nosehooverchain": nvt.NoseHooverChain,
    "nosehoovermetadynamics": nvt.NoseHooverMetadynamics,
    "batchednosehoovermetadynamics": nvt.BatchNoseHooverMetadynamics
}

BATCHED_OPT = {"BFGS": BatchedBFGS,
               "LBFGS": BatchedLBFGS}


def get_key(iroot, num_states):
    """
    Get energy key for the state of interest.
    Args:
        iroot (int): state of interest
        num_states (int): total number of states
    Returns:
        key (str): energy key
    """

    # energy if only one state
    if iroot == 0 and num_states == 1:
        key = "energy"

    # otherwise energy with state suffix
    else:
        key = "energy_{}".format(iroot)
    return key


def get_model(params):
    nn_id = params['nnid']
    # get the right weightpath (either regular or cluster-mounted)
    # depending on which exists
    weightpath = os.path.join(params['weightpath'], str(nn_id))
    if not os.path.isdir(weightpath):
        weightpath = os.path.join(params['mounted_weightpath'], str(nn_id))

    # get the model
    model = load_model(weightpath)

    return model


def get_non_mtd_idx(atoms):
    constraints = atoms.constraints
    fixed_idx = []
    for constraint in constraints:
        has_keys = False
        keys = ['idx', 'indices', 'index']
        for key in keys:
            if hasattr(constraint, key):
                val = np.array(getattr(constraint, key)
                               ).reshape(-1).tolist()
                fixed_idx += val
                has_keys = True
        if not has_keys:
            print(("WARNING: MTD indices not ignored for any atoms in constraint "
                   "%s; do not know how to find its fixed indices." % constraint))

    if not fixed_idx:
        return

    fixed_idx = np.array(list(set(fixed_idx)))

    return fixed_idx


def update_with_exclude(params,
                        pushing_params,
                        exclude_from_rmsd):

    idx = exclude_from_rmsd.get("idx")
    if idx is None:
        return pushing_params

    template_smiles = exclude_from_rmsd.get("template_smiles")
    if template_smiles is not None:
        assert "smiles" in params, ("Specified template_smiles in "
                                    "`exclude_from_rmsd`; need smiles in main "
                                    "params")
        substruc_idx = get_substruc_idx(template_smiles=template_smiles,
                                        smiles=params["smiles"])
        idx = substruc_idx[np.array(idx)].tolist()

    exclude_atoms = pushing_params.get("exclude_atoms")
    if exclude_atoms is None:
        exclude_atoms = []
    else:
        if hasattr(exclude_atoms, 'tolist'):
            exclude_atoms = exclude_atoms.tolist()

    exclude_atoms += idx
    pushing_params["exclude_atoms"] = list(set(exclude_atoms))

    return pushing_params


def get_pushing_params(params, atoms):

    sampling_dic = params["enhanced_sampling"]
    pushing_params = sampling_dic["params"]["pushing_params"]

    fixed_idx = get_non_mtd_idx(atoms)
    if fixed_idx is not None:
        pushing_params["exclude_atoms"] = fixed_idx

    # fixed atoms can also be specified in main params
    exclude_from_rmsd = params.get("exclude_from_rmsd")
    if exclude_from_rmsd is not None:
        pushing_params = update_with_exclude(params=params,
                                             pushing_params=pushing_params,
                                             exclude_from_rmsd=exclude_from_rmsd)

    return pushing_params


def init_calculator(atoms,
                    params,
                    model):
    """
    Set the calculator for the atoms and
    get the model.
    Args:
        atoms (AtomsBatch): atoms for geom of interest
        params (dict): dictionary of parameters
    Returns:
        model (nn.Module): nnpotential model
        en_key (str): energy key
    """

    opt_state = params.get("iroot", 0)
    num_states = params.get("num_states", 1)

    if 'en_key' in params:
        en_key = params['en_key']
    else:
        en_key = get_key(iroot=opt_state, num_states=num_states)

    # get and set the calculator

    if params.get("enhanced_sampling") is not None:
        sampling_dic = params["enhanced_sampling"]
        method = sampling_dic.get("method")

        lower_method = method.lower()
        upper_method = method.upper()

        if upper_method == "GAMD":
            nff_ase = NeuralGAMD(
                model=model,
                device=params.get('device', 'cuda'),
                en_key=en_key,
                model_kwargs=params.get("model_kwargs"),
                **sampling_dic["params"]
            )

        elif lower_method in ["nosehoovermetadynamics",
                              "batchednosehoovermetadynamics"]:
            # get the pushing params and also add the excluded atoms based on
            # what's being constrained
            pushing_params = get_pushing_params(params=params,
                                                atoms=atoms)

            if lower_method == "nosehoovermetadynamics":
                mtd_class = NeuralMetadynamics
            elif lower_method == "batchednosehoovermetadynamics":
                mtd_class = BatchNeuralMetadynamics

            nff_ase = mtd_class(
                model=model,
                device=params.get('device', 'cuda'),
                en_key=en_key,
                model_kwargs=params.get("model_kwargs"),
                directed=params["directed"],
                pushing_params=pushing_params
            )

        else:
            raise NotImplementedError

    else:
        nff_ase = NeuralFF(
            model=model,
            device=params.get('device', 'cuda'),
            en_key=en_key,
            model_kwargs=params.get("model_kwargs")
        )

    atoms.set_calculator(nff_ase)

    return en_key


def correct_hessian(restart_file,
                    hessian):
    """
    During an optimization, replace the approximate BFGS
    Hessian with the analytical nnpotential Hessian.
    Args:
        restart_file (str): name of the json file
            for restarting the optimization.
        hessian (list): Hessian
    Returns:
        None
    """

    # get the parameters from the restart file

    with open(restart_file, "r") as f:
        restart = read_json(f)

    new_restart = [hessian]
    new_restart += [i for i in restart[1:]]
    new_restart = tuple(new_restart)

    # save the restart file

    with open(restart_file, "w") as f:
        write_json(f, new_restart)


def get_output_keys(model):

    atomwisereadout = model.atomwisereadout
    # get the names of all the attributes of the readout dict
    readout_attr_names = dir(atomwisereadout)

    # restrict to the attributes that are ModuleDicts
    readout_dict_names = [name for name in readout_attr_names if
                          type(getattr(atomwisereadout, name)) is ModuleDict]

    # get the ModuleDicts
    readout_dicts = [getattr(atomwisereadout, name)
                     for name in readout_dict_names]

    # get their keys
    output_keys = [key for dic in readout_dicts for key in dic.keys()]

    return output_keys


def get_loader(model,
               nxyz_list,
               num_states,
               cutoff):

    base_keys = get_output_keys(model)
    grad_keys = [key + "_grad" for key in base_keys]

    ref_quant = [0] * len(nxyz_list)
    ref_quant_grad = [
        np.zeros(((len(nxyz_list[0])), 3)).tolist()] * len(nxyz_list)

    props = {"nxyz": nxyz_list}
    props.update({key: ref_quant for key in base_keys})
    props.update({key: ref_quant_grad for key in grad_keys})

    dataset = Dataset(props.copy())
    dataset.generate_neighbor_list(cutoff)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_dicts)

    return model, loader


def check_convg(atoms,
                restart_file):

    results = hessian_and_modes(ase_atoms=atoms)
    # Hessian in ASE units
    hessian = (np.array(results["hessianmatrix"]) /
               (EV_TO_AU * BOHR_RADIUS ** 2))
    # Frequencies in cm^(-1)
    freqs = results["vibfreqs"]

    neg_freqs = list(filter(lambda x: x < 0, freqs))
    num_neg = len(neg_freqs)
    if num_neg != 0:
        print(("Found {} negative frequencies; "
               "restarting optimization.").format(num_neg))
        correct_hessian(restart_file=restart_file,
                        hessian=hessian)

        # Give a kick in the direction of the lowest-frequency
        # imaginary mode. This not only helps the optimization
        # by going in the right direction, but without it
        # the optimization would end at the first step, because
        # the gradient would be within the tolerance.

        # Also, the kick means that the saved Hessian is not
        # exactly right, because it's actually the Hessian from
        # before the kick. But it's still a better estimate than
        # the diagonal matrix estimate that would normally be used
        # in an optimizer

        # Note that fmax should be quite small if you're going to check
        # for negative frequencies and then give a kick, because with
        # non-negligible forces and a small imaginary frequency, the
        # resulting energy change can actually be positive

        nxyz = np.concatenate([atoms.get_atomic_numbers().reshape(-1, 1),
                               atoms.get_positions()], axis=-1)

        nxyz_bohrs = init_displace(eigvecs=np.array(results['modes']),
                                   freqs_cm=np.array(results['vibfreqs']),
                                   # expected change of 0.3 kcal / mol
                                   init_displ_de=0.00047,
                                   ts_nxyz=nxyz,
                                   mode=-1)

        ens = []
        pos_set = []

        for nxyz_bohr in nxyz_bohrs:
            new_pos = nxyz_bohr[:, 1:] * BOHR_RADIUS
            pos_set.append(new_pos)

            atoms.set_positions(new_pos)
            ens.append(atoms.get_potential_energy())

        ens = np.array(ens).reshape(-1)
        atoms.set_positions(pos_set[np.argmin(ens)])

        return False

    else:
        print(("Found no negative frequencies; "
               "optimization complete."))

        return True


def get_opt_kwargs(params, nbr_update_period):

    # params with the right name for max_step
    new_params = copy.deepcopy(params)
    new_params["steps"] = nbr_update_period

    opt_kwargs = {key: val for key,
                  val in new_params.items() if key in OPT_KEYS}

    return opt_kwargs


def get_opt_module(atoms,
                   params):

    opt_name = params.get("opt_type", "BFGS")
    # if you have multiple molecules in a batch, you'll need an NFF custom batched
    # optimizer

    num_atoms = atoms.num_atoms
    if num_atoms.shape[0] > 1:
        assert opt_name in BATCHED_OPT, ("No custom NFF batched implementation of the "
                                         "%s optimizer!" % opt_name)
        opt_module = BATCHED_OPT[opt_name]
    else:
        opt_module = getattr(optimize, opt_name)

    return opt_module


def opt_conformer(atoms,
                  params,
                  model):

    converged = False
    max_rounds = params.get("max_rounds", MAX_ROUNDS)
    check_hess = params.get("check_hess", True)
    max_steps = params.get("opt_max_step", 500)
    nbr_update_period = params.get("nbr_list_update_freq", 1)

    if max_rounds == 1:
        restart_file = None
    else:
        restart_file = RESTART_FILE

    for iteration in range(max_rounds):

        opt_module = get_opt_module(atoms=atoms,
                                    params=params)
        opt_kwargs = get_opt_kwargs(params=params,
                                    nbr_update_period=nbr_update_period)

        # only use the restart file after the first iteration, or else
        # it'll use restart files from other conformers / jobs

        if iteration == 0 and os.path.isfile(str(restart_file)):
            os.remove(restart_file)

        # replace enhanced sampling potential with regular potential, if necessary
        if params.get("enhanced_sampling") is not None:

            regular_params = copy.deepcopy(params)
            method = params["enhanced_sampling"].get("method")

            # proper potential and also reset hydrogen mass
            if 'metadynamics' in method.lower():
                regular_params['md_type'] = "NoseHoover"
                keep_constraints = [i for i in atoms.constraints if not
                                    isinstance(i, FixInternals) and not
                                    isinstance(i, FixBondLengths)]
                atoms.constraints = keep_constraints
                reset_h_mass(atoms)

            if "enhanced_sampling" in regular_params:
                regular_params.pop("enhanced_sampling")

            init_calculator(atoms,
                            regular_params,
                            model=model)

        total_steps = 0
        dyn = opt_module(atoms,
                         restart=restart_file)

        while total_steps < max_steps:
            dyn_converged = dyn.run(**opt_kwargs)

            if dyn_converged:
                break

            atoms.update_nbr_list()
            opt_kwargs['steps'] += nbr_update_period
            total_steps += nbr_update_period

        if check_hess:
            hess_converged = check_convg(atoms=atoms,
                                         restart_file=restart_file)
        else:
            hess_converged = True

        if dyn_converged and hess_converged:
            converged = True
            break

    return atoms, converged


def get_confs(params,
              traj_filename,
              thermo_filename,
              num_starting_poses,
              model):

    with open(thermo_filename, "r") as f:
        lines = f.readlines()
    energies = []
    for line in lines:
        try:
            energies.append(float(line.split()[2]))
        except ValueError:
            pass

    sort_idx = np.argsort(energies)

    trj = AseTrajectory(traj_filename)
    sample_by_en = params.get("sample_by_energy", True)

    if sample_by_en:
        sorted_steps = np.array(range(len(lines)))[
            sort_idx[:num_starting_poses]]
    else:
        all_steps = np.array(range(len(trj)))
        denom = max([(num_starting_poses - 1), 1])
        num_skip = int(np.floor((len(all_steps) - 1) /
                                denom))
        sorted_steps = all_steps[::num_skip][:num_starting_poses]

    best_confs = []

    for i in sorted_steps:
        atoms = trj[i]

        these_params = copy.deepcopy(params)
        these_params['nxyz'] = np.concatenate([
            atoms.get_atomic_numbers().reshape(-1, 1),
            atoms.get_positions()],
            axis=-1)

        atoms_batch = params_to_atoms(these_params,
                                      model=model)
        best_confs.append(atoms_batch)

    return best_confs


def get_nve_params(params):

    nve_params = copy.deepcopy(nve.DEFAULTNVEPARAMS)
    common_keys = [key for key in nve_params.keys() if key in params]

    for key in common_keys:
        nve_params[key] = params[key]

    integrator = params["integrator"]
    if type(integrator) is str:
        integ_name = integrator.lower().replace("_", "")
        nve_params["integrator"] = INTEGRATOR_DIC[integ_name]

    return nve_params


def get_trj_file(params):
    trj_file = params.get("traj_filename")
    if trj_file is None:
        trj_file = nve.DEFAULTNVEPARAMS["traj_filename"]
    return trj_file


def get_log_file(params):
    thermo_filename = params.get("thermo_filename")
    if thermo_filename is None:
        thermo_filename = nve.DEFAULTNVEPARAMS["thermo_filename"]
    return thermo_filename


def pre_clean(params):
    thermo_filename = get_log_file(params)
    trj_file = get_trj_file(params)

    if os.path.isfile(thermo_filename):
        os.remove(thermo_filename)

    if os.path.isfile(trj_file):
        os.remove(trj_file)


def get_non_nve(atoms,
                params,
                reload_ground=True):

    pre_clean(params)
    logfile = get_log_file(params)
    trj_file = get_trj_file(params)

    nbr_update_period = params.get("nbr_list_update_freq", 1)

    # note - there's no temperature scaling here for any
    # fixed atoms

    params.update({"logfile": logfile,
                   "trajectory": trj_file,
                   "nbr_update_period": nbr_update_period,
                   "max_steps": params.get("steps"),
                   "timestep": params.get("time_step")})

    method = METHOD_DIC[params["md_type"].lower()]
    dynamics = method(atoms, **params)

    return dynamics


def get_nve(atoms,
            params):

    pre_clean(params)

    nve_params = get_nve_params(params)
    nve_instance = nve.Dynamics(atomsbatch=atoms,
                                mdparam=nve_params)

    return nve_instance


def md_to_conf(atoms,
               params,
               model):

    md_type = params.get("md_type", "nve")
    func = get_nve if (md_type == "nve") else get_non_nve
    dynamics = func(atoms=atoms, params=params)

    dynamics.run()

    num_starting_poses = params.get("num_starting_poses", NUM_CONFS)
    thermo_filename = get_log_file(params)
    trj_file = get_trj_file(params)

    best_confs = get_confs(params=params,
                           traj_filename=trj_file,
                           thermo_filename=thermo_filename,
                           num_starting_poses=num_starting_poses,
                           model=model)

    return best_confs


def confs_to_opt(params,
                 best_confs,
                 model,
                 return_sorted=True,
                 ref_idx=0):

    convg_atoms = []
    energy_list = []

    assert_converged = params.get("assert_converged", True)

    for i in range(len(best_confs)):
        atoms = copy.deepcopy(best_confs[i])
        print("Optimizing conformer %d" % (ref_idx + i + 1))

        restart_file = RESTART_FILE
        if os.path.isfile(restart_file):
            os.remove(restart_file)

        atoms, converged = opt_conformer(atoms=atoms,
                                         params=params,
                                         model=model)

        if converged or (not assert_converged):
            convg_atoms.append(atoms)
            energy_list.append(atoms.get_potential_energy())

    if (not convg_atoms) and assert_converged:
        raise Exception("No successful optimizations")

    if return_sorted:
        # sort results by energy
        best_idx = np.argsort(np.array(energy_list).reshape(-1))

        sorted_ens = np.array(energy_list)[best_idx].reshape(-1).tolist()
        sorted_atoms = [convg_atoms[i] for i in best_idx]

    else:
        sorted_ens = np.array(energy_list).reshape(-1).tolist()
        sorted_atoms = convg_atoms

    if params.get("do_save", True):
        trj_writer = TrajectoryWriter(filename=OPT_FILENAME)
        for atoms in sorted_atoms:
            trj_writer.write(atoms)
        trj_writer.close()

        with open(EN_PATH, 'wb') as f:
            pickle.dump(sorted_ens, f)

        print("{} confs written".format(len(sorted_atoms)))

    return sorted_atoms, sorted_ens


def get_params(info_file):
    with open(info_file) as f:
        info = json.load(f)
    non_details = {key: val for key, val in info.items(
    ) if key != "details"}

    params = {}
    params.update(non_details)

    if 'details' in info:
        params.update(info['details'])

    return params


def check_constraints(name,
                      idx,
                      vals):

    use_name = name.split("_")[0]
    if use_name.endswith("s"):
        use_name = use_name[:-1]

    msg = "Supplied indices for %s constraints but not values" % use_name
    assert vals is not None, msg

    assert len(idx) == len(vals), ("Index length %d"
                                   " doesn't match value length %d" % (
                                       len(idx), len(vals)))


def make_constraints(name,
                     idx,
                     vals):

    check_constraints(name=name,
                      idx=idx,
                      vals=vals)

    constraints = []
    for idx, val in zip(idx, vals):
        # note that angles that are exactly 180 or 360 can cause
        # problems

        constraint = [val, idx]
        constraints.append(constraint)

    return constraints


def get_constraint_vals(atoms,
                        name,
                        idx,
                        init_vals):
    vals = []
    for these_idx, this_val in zip(idx, init_vals):
        if this_val != 'auto':
            vals.append(this_val)
            continue

        if name == 'bonds':
            all_pos = atoms.get_positions()
            bond_pos = all_pos[np.array(these_idx)]
            new_val = np.linalg.norm(bond_pos[0] - bond_pos[1])

        elif name == 'dihedrals_deg':
            new_val = atoms.get_dihedral(*these_idx)

        elif name == 'angles_deg':
            new_val = atoms.get_angle(*these_idx)

        else:
            raise NotImplementedError

        vals.append(new_val)

    return vals


def constrain_internals(atoms,
                        bond_idx=None,
                        angle_idx=None,
                        dihed_idx=None,
                        bond_lengths=None,
                        angles=None,
                        diheds=None,
                        **kwargs):

    idx_dic = {"bonds": [bond_lengths, bond_idx],
               "angles_deg": [angles, angle_idx],
               "dihedrals_deg": [diheds, dihed_idx]}

    constraint_dic = {}

    for name, lst in idx_dic.items():
        if all([i is None for i in lst]):
            continue

        idx = lst[1]
        vals = get_constraint_vals(atoms=atoms,
                                   name=name,
                                   idx=idx,
                                   init_vals=lst[0])
        constraints = make_constraints(name=name,
                                       idx=idx,
                                       vals=vals)

        constraint_dic[name] = constraints

    c = FixInternals(bonds=constraint_dic.get("bonds"),
                     angles_deg=constraint_dic.get("angles_deg"),
                     dihedrals_deg=constraint_dic.get("dihedrals_deg"),
                     epsilon=1e-3)

    atoms.constraints.append(c)


def constrain_hookean(atoms,
                      kwargs):

    for constrain_type, params in kwargs.items():
        idx = params["idx"]
        if idx is None:
            continue

        force_consts = params["force_consts"]
        if isinstance(force_consts, list):
            force_consts = np.array(force_consts).astype('float')
        # convert to eV for dynamics

        force_consts /= EV_TO_KCAL_MOL

        pos = atoms.get_positions()

        if constrain_type == "angles":
            c = ConstrainAngles(idx=idx,
                                atoms=atoms,
                                force_consts=force_consts,
                                targ_angles=params.get("targets"))
            constraints = [c]

        elif constrain_type == "dihedrals":
            c = ConstrainDihedrals(idx=idx,
                                   atoms=atoms,
                                   force_consts=force_consts,
                                   targ_diheds=params.get("targets"))
            constraints = [c]

        elif constrain_type == "bonds":
            c = ConstrainBonds(idx=idx,
                               atoms=atoms,
                               force_consts=force_consts,
                               targ_lengths=params.get("targets"))
            constraints = [c]

        elif constrain_type == "atoms":
            constraints = []
            for idx_list in idx:
                for i, this_idx in enumerate(idx_list):
                    if isinstance(force_consts, list):
                        k = force_consts[i]
                    else:
                        k = force_consts

                    epsilon = 0  # 1e-3
                    c = Hookean(a1=this_idx,
                                a2=(pos[this_idx] + epsilon),
                                k=k,
                                rt=0)
                    constraints.append(c)

        else:
            raise NotImplementedError

        atoms.constraints += constraints


def fix_atoms(atoms, idx):
    c = FixAtoms(indices=idx)
    atoms.constraints.append(c)


def coords_to_xyz(coords):
    n = [PERIODICTABLE.GetAtomicNumber(dic['element'])
         for dic in coords]
    xyz = [[dic['x'], dic['y'], dic['z']] for dic
           in coords]

    nxyz = np.concatenate([np.array(n).reshape(-1, 1),
                           np.array(xyz)], axis=-1)

    return nxyz


def get_substruc_idx(template_smiles,
                     smiles):

    template_mol = Chem.MolFromSmiles(template_smiles)
    this_mol = Chem.MolFromSmiles(smiles)

    idx_tuple = this_mol.GetSubstructMatch(template_mol)
    substruc_idx = np.array(idx_tuple)

    return substruc_idx


def translate_sub_dic_template(smiles,
                               sub_dic):

    template_smiles = sub_dic.get("template_smiles")
    idx = sub_dic.get("idx")

    if template_smiles is None or idx is None:
        return sub_dic

    msg = ("Tried to make constraints from template smiles, but smiles of the "
           "current species is not given!")
    assert smiles is not None, msg

    substruc_idx = get_substruc_idx(template_smiles=template_smiles,
                                    smiles=smiles)

    idx = np.array(idx)
    base_shape = idx.shape
    new_idx = substruc_idx[idx.reshape(-1)].reshape(*base_shape).tolist()
    sub_dic['idx'] = new_idx
    sub_dic.pop('template_smiles')

    return sub_dic


def translate_template_recursive(smiles,
                                 dic):

    dic = translate_sub_dic_template(smiles=smiles,
                                     sub_dic=dic)
    for key, val in dic.items():
        if isinstance(val, dict):
            translate_template_recursive(smiles=smiles,
                                         dic=val)
            dic[key] = val


def translate_template(params):
    """
    Translate any constraints that are done with respect to a template SMILES with
    known indices.
    """

    new_params = copy.deepcopy(params)
    smiles = new_params.get('smiles')

    fixed_dic = new_params.get("fixed_atoms")
    constraints = new_params.get("constraints")

    if fixed_dic is None:
        fixed_dic = {}
    if constraints is None:
        constraints = {}

    translate_template_recursive(smiles=smiles,
                                 dic=fixed_dic)
    translate_template_recursive(smiles=smiles,
                                 dic=constraints)

    new_params['constraints'] = constraints
    new_params['fixed_atoms'] = fixed_dic

    return new_params


def get_bond_idx(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    bond_idx = np.array([[i.GetBeginAtomIdx(), i.GetEndAtomIdx()] for
                         i in mol.GetBonds()])

    return bond_idx


def increase_h_mass(atoms):
    atomic_nums = atoms.get_atomic_numbers()
    idx = atomic_nums == 1
    masses = atoms.get_masses()
    masses[idx] = 2.0

    atoms.set_masses(masses)


def reset_h_mass(atoms):
    atomic_nums = atoms.get_atomic_numbers()
    idx = atomic_nums == 1
    masses = atoms.get_masses()
    masses[idx] = 1.008

    atoms.set_masses(masses)


def add_shake_atoms(atoms,
                    params):

    assert "smiles" in params, "Need to specify SMILES to constrain bonds with SHAKE"
    bond_idx = get_bond_idx(smiles=params["smiles"])
    c = FixBondLengths(pairs=bond_idx,
                       # it doesn't converge when maxiter = 500 and the tolerance is the
                       # default of 1e-13
                       tolerance=1e-7)

    atoms.constraints.append(c)


def add_all_constraints(atoms,
                        params,
                        do_translate=True):

    if do_translate:
        these_params = translate_template(params)
    else:
        these_params = params

    fixed_idx = these_params.get("fixed_atoms", {}).get("idx")
    constraints = these_params.get("constraints")
    device = these_params.get('device', 0)

    if constraints is not None:

        if "hookean" in constraints:
            kwargs = constraints["hookean"]
            constrain_hookean(atoms=atoms,
                              kwargs=kwargs)

        remain_constraints = {key: val for key, val
                              in constraints.items() if key != 'hookean'}

        if remain_constraints:
            constrain_internals(atoms=atoms,
                                device=device,
                                **remain_constraints)

    # if you ask for metadynamics, then the bond lengths all need to be held fixed
    # with SHAKE, and the hydrogen mass should be set to 2 amu

    enhanced_sampling = these_params.get("enhanced_sampling", {})
    enhanced_method = enhanced_sampling.get("method", "")

    if "metadynamics" in enhanced_method.lower():
        # shake is really slow, so it doesn't help by letting you use a larger time
        # step, and from what I've seen it's not necessary for keeping the molecule
        # together in metadynamics. So by default we won't use it

        add_shake = enhanced_sampling.get("shake", False)
        if add_shake:
            add_shake_atoms(atoms=atoms,
                            params=params)

        # Double the hydrogen mass - this makes them move more slowly and actually
        # does let us get away with a bigger time step
        increase_h_mass(atoms)

    # do this after any internal constraints to guarantee zero forces on these atoms
    if fixed_idx is not None:
        fix_atoms(atoms=atoms, idx=fixed_idx)


def params_to_atoms(params,
                    model):

    if 'nxyz' in params:
        nxyz = np.array(params['nxyz'])
    else:
        nxyz = coords_to_xyz(params["coords"])

    device = params.get("device", 0)
    atoms = AtomsBatch(nxyz[:, 0],
                       nxyz[:, 1:],
                       device=device,
                       cutoff=params["cutoff"],
                       requires_large_offsets=params["requires_large_offsets"],
                       cutoff_skin=params["cutoff_skin"],
                       directed=params["directed"])

    # this has to come before `init_calculator` so that `init_calculator` can look
    # at the constraints and decide not to add them to a possible RMSD-based mTD
    # term

    add_all_constraints(atoms=atoms,
                        params=params)

    init_calculator(atoms=atoms,
                    params=params,
                    model=model)

    directed = (not any([isinstance(model, i) for i in UNDIRECTED]))
    atoms.directed = directed

    return atoms


def opt_from_atoms(atoms,
                   params,
                   model=None):

    do_md = params.get("do_md", True)
    if model is None:
        model = get_model(params)

    if do_md:
        best_confs = md_to_conf(atoms=atoms,
                                params=params,
                                model=model)
    else:
        best_confs = [atoms]

    best_atoms, sorted_ens = confs_to_opt(params=params,
                                          best_confs=best_confs,
                                          model=model)

    return best_atoms, sorted_ens


def opt_from_params(params,
                    model=None):

    # Important to not re-load the model over and over again - do it once
    # at the beginning and then re-use it

    if model is None:
        model = get_model(params)

    atoms = params_to_atoms(params,
                            model=model)
    best_atoms, sorted_ens = opt_from_atoms(atoms=atoms,
                                            params=params,
                                            model=model)

    return best_atoms, sorted_ens


def main():
    parser = argparse.ArgumentParser(
        description="optimize a structure with a neural potential")
    parser.add_argument('info_file', type=str, default=DEFAULT_INFO_FILE,
                        help="file containing all parameters")

    args = parser.parse_args()
    params = get_params(info_file=args.info_file)
    opt_from_params(params)


if __name__ == "__main__":
    main()

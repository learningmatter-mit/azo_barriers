"""
Script for doing neural_confgen with inference performed on multiple molecules
in the same batch. This is much faster than running N copies of neural_confgen.py
in parallel for N different molecules, which seems to have a maximum speedup of
N / (time / (time for single molecule)) = 2.5. That could be related to the GPU having
to continually allocate memory for the different processes and hence slowing down.

"""

import os
import copy
import torch
import json
import time
import numpy as np
import shutil
from rdkit import Chem
import sys

from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.calculators.calculator import Calculator

from nff.io.ase import AtomsBatch, UNDIRECTED
from nff.utils import constants as const

from barriers.utils.neuraloptimizer import (translate_template,
                                            add_all_constraints,
                                            init_calculator,
                                            coords_to_xyz,
                                            get_non_nve,
                                            get_model,
                                            confs_to_opt,
                                            get_trj_file,
                                            OPT_FILENAME,
                                            update_with_exclude)

from barriers.confgen.neural_confgen import (update_params,
                                             get_params,
                                             JSON_KEYS,
                                             parse_args,
                                             ARGS_PATH,
                                             get_num_starting_poses,
                                             parse_path,
                                             make_xyz_text,
                                             make_rand_string,
                                             bash_command,
                                             read_unique,
                                             get_is_done,
                                             move_general_file,
                                             FINAL_OPT_FILENAME,
                                             summarize_final,
                                             time_to_msg,
                                             set_mtd_time)

PERIODICTABLE = Chem.GetPeriodicTable()

# for identifying slow-to-converge conformers, so that we can group them together
# and continue their optimization, instead of having one conformer in a batch take
# forever while the others are already done
INIT_MAX_ITER = {"coarse": [20, 30, 40, 50, 60],
                 "tight": [60, 75, 90, 105, 120],
                 "vtight": [200, 275, 350, 425, 500]}

MAX_RETRIES = 1
LOG_FILE = "neural_confgen.log"


class StorageCalculator(Calculator):
    def __init__(self,
                 implemented_properties,
                 results,
                 atoms):

        self.implemented_properties = implemented_properties
        self.results = results
        self.atoms = atoms
        self.name = "StorageCalculator"
        self.parameters = {}

    def calculate():
        return


def get_translated_params(params_list):
    """
    Translate the constraint parameters of each molecule -- e.g. if it uses
    indices from a template molecule, translate those to indices of the current
    molecule.
    """

    new_params_list = []
    for params in params_list:
        these_params = translate_template(params)
        new_params_list.append(these_params)

    return new_params_list


def get_exclude_idx(params_list):

    increment = 0
    all_exclude_idx = []

    for base_params in params_list:
        params = copy.deepcopy(base_params)
        sampling_dic = params.get("enhanced_sampling")
        if sampling_dic is not None:
            pushing_params = sampling_dic["params"]["pushing_params"]
            exclude_from_rmsd = params.get("exclude_from_rmsd")

            if exclude_from_rmsd is None:
                exclude_idx = None
            else:
                pushing_params['exclude_atoms'] = None
                update_with_exclude(params=params,
                                    pushing_params=pushing_params,
                                    exclude_from_rmsd=exclude_from_rmsd)
                exclude_idx = pushing_params.get("exclude_atoms")

            if exclude_idx is not None:
                all_exclude_idx += (np.array(exclude_idx) + increment).tolist()

        increment += len(base_params.get('nxyz', base_params.get('coords')))

    return all_exclude_idx


def increment_translated_params(these_params,
                                molecule_num,
                                num_atoms):
    """
    Increment any fixed or constrained indices by sum(number of atoms in
    previous molecules) for a molecule in the batch
    """

    increment = sum(num_atoms[:molecule_num])

    fixed_idx = these_params.get("fixed_atoms", {}).get("idx")
    if fixed_idx is not None:
        these_params['fixed_atoms']['idx'] = [i + increment for i in fixed_idx]

    constraints = these_params.get("constraints")

    if constraints is not None:

        if "hookean" in constraints:
            kwargs = constraints["hookean"]

            for constrain_type, params in kwargs.items():
                idx = params.get("idx")
                if idx is None:
                    continue
                params['idx'] = [[i + increment for i in these_idx]
                                 for these_idx in idx]

        remain_constraints = {key: val for key, val
                              in constraints.items() if key != 'hookean'}

        if remain_constraints:
            keys = ['bond_idx', 'angle_idx', 'dihed_idx']
            for key in keys:
                if remain_constraints.get(key) is None:
                    continue
                idx = remain_constraints[key]
                constraints[key] = [i + increment for i in idx]

    enhanced_sampling = these_params.get("enhanced_sampling", {})
    enhanced_method = enhanced_sampling.get("method", "")
    if "metadynamics" in enhanced_method.lower():
        # shake is really slow, so it doesn't help by letting you use a larger time
        # step, and from what I've seen it's not necessary for keeping the molecule
        # together in metadynamics. So by default we won't use it

        add_shake = enhanced_sampling.get("shake", False)
        assert not add_shake, "Batching not implemented with SHAKE constraints"


def combine_params_list(new_params_list):
    final_params = copy.deepcopy(new_params_list[0])

    for these_params in new_params_list[1:]:
        fixed_idx = these_params.get("fixed_atoms", {}).get("idx")
        if fixed_idx is not None:
            if 'fixed_atoms' not in final_params:
                final_params['fixed_atoms'] = {"idx": []}
            elif final_params['fixed_atoms'].get("idx") is None:
                final_params['fixed_atoms']['idx'] = []

            final_params['fixed_atoms']['idx'] += fixed_idx

        constraints = these_params.get("constraints")

        if constraints is not None:

            if "hookean" in constraints:
                kwargs = constraints["hookean"]

                for constrain_type, params in kwargs.items():
                    idx = params.get("idx")
                    if idx is None:
                        continue

                    ref_params = final_params['constraints']['hookean'][constrain_type]

                    these_force_consts = params['force_consts']
                    ref_force_consts = ref_params['force_consts']

                    if not isinstance(these_force_consts, list):
                        these_force_consts = [these_force_consts] * len(idx)
                    if not isinstance(ref_force_consts, list):
                        ref_force_consts = ([ref_force_consts] *
                                            len(ref_params['idx']))

                    ref_params['force_consts'] = (ref_force_consts +
                                                  these_force_consts)
                    ref_params['idx'] += idx
                    ref_params['targets'] += params['targets']

            remain_constraints = {key: val for key, val
                                  in constraints.items() if key != 'hookean'}

            if remain_constraints:
                keys = ['bond_idx', 'angle_idx', 'dihed_idx']
                for key in keys:
                    if remain_constraints.get(key) is None:
                        continue
                    idx = remain_constraints[key]
                    final_params['constraints'][key]['idx'] += idx

    return final_params


def update_combined_w_exclude(combined_params,
                              params_list):
    all_exclude_idx = get_exclude_idx(params_list=params_list)
    combined_params['exclude_from_rmsd'] = {"idx": all_exclude_idx}

    return combined_params


def make_batched_constraint_dic(params_list):

    new_params_list = get_translated_params(params_list)
    num_atoms = [len(params.get('nxyz', params.get('coords')))
                 for params in new_params_list]

    # add all the increments
    for i, these_params in enumerate(new_params_list):
        increment_translated_params(these_params=these_params,
                                    molecule_num=i,
                                    num_atoms=num_atoms)

    # now combine together
    combined_params = combine_params_list(new_params_list)

    # add any indices to exclude from RMSD
    combined_params = update_combined_w_exclude(combined_params=combined_params,
                                                params_list=params_list)

    return combined_params


def add_batched_constraints(atoms_batch,
                            params_list):

    params = make_batched_constraint_dic(params_list)
    add_all_constraints(atoms=atoms_batch,
                        params=params,
                        # already translated above
                        do_translate=False)

    return params


def params_list_to_nxyz(params_list):
    nxyz_list = []
    for params in params_list:
        if 'nxyz' in params:
            nxyz = torch.Tensor(params['nxyz'])
        else:
            nxyz = torch.Tensor(coords_to_xyz(params["coords"]))
        nxyz_list.append(nxyz)

    return nxyz_list


def final_exclude_update(combined_params,
                         use_params):

    exclude = combined_params.get('exclude_from_rmsd')
    if exclude is None:
        return

    use_params["exclude_from_rmsd"] = exclude


def params_to_atoms_batch(params_list,
                          model):
    """
    Combine different molecules into one big `atoms_batch` object, with the constraints
    added to the atoms with the correct indices.
    """

    nxyz_list = params_list_to_nxyz(params_list)
    nxyz = torch.cat([nxyz for nxyz in nxyz_list])

    # `num_atoms` so we can keep track of which molecule is which
    num_atoms = torch.LongTensor([len(i) for i in nxyz_list])
    props = {"num_atoms": num_atoms}

    # copy.deepcopy is important so that the batched `exclude_atoms` don't get
    # associated with `params_list[0]`
    params = copy.deepcopy(params_list[0])
    device = params.get("device", 0)
    atoms_batch = AtomsBatch(nxyz[:, 0],
                             nxyz[:, 1:],
                             device=device,
                             cutoff=params["cutoff"],
                             requires_large_offsets=params["requires_large_offsets"],
                             cutoff_skin=params["cutoff_skin"],
                             directed=params["directed"],
                             props=props)

    # this has to come before `init_calculator` so that `init_calculator` can look
    # at the constraints and decide not to add them to a possible RMSD-based mTD
    # term

    combined_params = add_batched_constraints(atoms_batch=atoms_batch,
                                              params_list=params_list)

    final_exclude_update(combined_params=combined_params,
                         use_params=params)

    init_calculator(atoms=atoms_batch,
                    params=params,
                    model=model)

    directed = (not any([isinstance(model, i) for i in UNDIRECTED]))
    atoms_batch.directed = directed

    return atoms_batch


def fix_batch_keys(params):
    md_type_lower = params["md_type"].lower()
    method_lower = params['enhanced_sampling']['method'].lower()

    new_vals = []

    for val in [md_type_lower, method_lower]:
        if "batch" in val:
            new_vals.append(val)
            continue

        if val == "nosehoovermetadynamics":
            new_val = "BatchedNoseHooverMetadynamics"
        else:
            raise NotImplementedError

        new_vals.append(new_val)

    md_type, method = new_vals
    params["md_type"] = md_type
    params['enhanced_sampling']['method'] = method


def homogenize_time_steps(params_list):
    argmax = np.argmax([params['time_step'] for params in params_list])
    ref_params = params_list[argmax]

    time_step = ref_params['time_step']

    for params in params_list:
        params.update({"time_step": time_step})


def order_by_time(params_list):
    times = np.array([params['mtd_time'] for params in params_list])
    # do the biggest ones last
    idx = np.argsort(times)
    new_params_list = []

    for mol_index, i in enumerate(idx):
        params_list[i]['mol_index'] = mol_index
        new_params_list.append(params_list[i])

    return new_params_list


def load_all_params(base_dir,
                    args):

    sub_folders = [os.path.join(base_dir, i)
                   for i in os.listdir(base_dir)]
    sub_folders = [folder for folder in sub_folders if
                   os.path.isfile(os.path.join(folder, 'job_info.json'))]

    params_list = []

    for i, sub_folder in enumerate(sub_folders):

        file = os.path.join(sub_folder, 'job_info.json')
        params = copy.deepcopy(args.__dict__)
        params.update(get_params(info_file=file))

        for key in JSON_KEYS:
            val = params.get(key)
            try:
                params[key] = json.loads(val)
            except TypeError:
                continue

        # Get parameters that needed to be specified in `neuraloptimizer` that are now
        # fixed in `neural_confgen`
        update_params(params=params)

        # Add "batched" to the MD type if necessary
        fix_batch_keys(params=params)

        params.update({"job_dir": sub_folder,
                       # also record the index of these parameters, which will be
                       # useful later on
                       "mol_index": i})
        params_list.append(params)

    # order by time
    params_list = order_by_time(params_list)

    # make the number of steps and time step the same for all
    homogenize_time_steps(params_list=params_list)

    return params_list


def get_first_atoms_batch(params_list,
                          num_parallel,
                          model):

    atoms_batch = params_to_atoms_batch(params_list=params_list[:num_parallel],
                                        model=model)

    return atoms_batch


def get_representative_params(this_params_list):

    times = [params["mtd_time"] for params in this_params_list]
    mtd_time_idx = np.argmax(times)
    rep_params = this_params_list[mtd_time_idx]
    # mtd_time in ps
    rep_params["steps"] = int(np.ceil(rep_params["mtd_time"] * 1000 /
                                      rep_params["time_step"]))

    return rep_params


def run_md(atoms_batch,
           base_dir,
           this_params_list,
           params):

    md_type = params["md_type"].lower()
    assert 'batch' in md_type, 'Need to request a batched integrator!'

    # even if it's batched NVE, it's not regular NVE, so we use the
    # general non-NVE dynamics-creating function

    dynamics = get_non_nve(atoms=atoms_batch,
                           params=params)

    dynamics.run()

    save_separate_trjs(base_dir=base_dir,
                       this_params_list=this_params_list)


def general_save_separate(base_dir,
                          this_params_list,
                          file_name):
    """
    Note -- `this_params_list` has to have everything in the same order as the
    atoms in the batch, so that we save things in the right directory
    """

    trj_path = os.path.join(base_dir, file_name)
    trj = Trajectory(trj_path)
    num_atoms = [len(params.get('nxyz', params.get('coords')))
                 for params in this_params_list]
    trjs = [[] for _ in range(len(num_atoms))]

    for combined_atoms in trj:

        pos = combined_atoms.get_positions()
        if np.bitwise_not(np.isfinite(pos)).any():
            continue

        # for saving separate energies and forces

        batched_ens = combined_atoms.get_potential_energy()
        batched_forces = combined_atoms.get_forces()
        split_forces = np.split(batched_forces, np.cumsum(num_atoms))[:-1]

        atoms_batch = AtomsBatch(combined_atoms,
                                 props={"num_atoms": torch.LongTensor(num_atoms)})
        atoms_list = atoms_batch.get_list_atoms()

        for i, atoms in enumerate(atoms_list):

            results = {"energy": np.array([batched_ens[i]]),
                       "forces": split_forces[i]}
            calc = StorageCalculator(implemented_properties=['energy', 'forces'],
                                     results=results,
                                     atoms=atoms)
            atoms.set_calculator(calc)
            trjs[i].append(atoms)

    assert len(this_params_list) == len(trjs)

    new_paths = []

    for params, trj in zip(this_params_list, trjs):
        save_dir = params["job_dir"]
        new_path = move_general_file(file=file_name,
                                     job_dir=save_dir)

        new_paths.append(new_path)

        save_path = os.path.join(save_dir, file_name)

        # Note - when the atoms are from a trajectory, the forces are computed
        # with the addition of RMSD-based pushing forces. When they are from
        # an optimization, they are not. In all cases, however, the forces include
        # the extra forces from any constraints applied

        writer = TrajectoryWriter(save_path)
        for atoms in trj:
            writer.write(atoms)
        writer.close()

        new_path = move_general_file(file=file_name,
                                     job_dir=save_dir)

        new_paths.append(new_path)

    return new_paths


def save_separate_trjs(base_dir,
                       this_params_list):

    params = this_params_list[0]
    file_name = get_trj_file(params)
    out = general_save_separate(base_dir=base_dir,
                                this_params_list=this_params_list,
                                file_name=file_name)
    return out


def save_separate_opts(base_dir,
                       this_params_list):

    out = general_save_separate(base_dir=base_dir,
                                this_params_list=this_params_list,
                                file_name=OPT_FILENAME)
    return out


def to_atoms_batch(confs,
                   this_params_list,
                   model):

    trj = []
    for conf in confs:
        atoms_batch = params_to_atoms_batch(params_list=this_params_list,
                                            model=model)
        atoms_batch.set_positions(conf.get_positions())
        trj.append(atoms_batch)

    return trj


def atoms_batch_to_nxyz(atoms_batch,
                        num_atoms=None):

    if num_atoms is None:
        num_atoms = atoms_batch.num_atoms

    pos_list = np.split(atoms_batch.get_positions(),
                        np.cumsum(num_atoms))[:-1]
    num_list = np.split(atoms_batch.get_atomic_numbers(),
                        np.cumsum(num_atoms))[:-1]

    nxyz_list = [np.concatenate([nums.reshape(-1, 1), pos],
                                axis=-1) for nums, pos in zip(num_list, pos_list)]

    return nxyz_list


def update_conf_dic_using_params(nxyz_lists,
                                 this_params_list,
                                 conf_dic,
                                 sampled_ens,
                                 stage,
                                 overwrite):

    sub_dic = conf_dic[stage]

    # over-write any old nxyzs, so we can use all nxyzs in this dictionary
    # to make atomsbatches for optimization

    if overwrite:
        for key in sub_dic.keys():
            sub_dic[key]['nxyz'] = []
            sub_dic[key]['energies'] = []

    for nxyz_list, energy_list in zip(nxyz_lists, sampled_ens):
        mol_indices = [params["mol_index"] for params in this_params_list]

        for i, nxyz in enumerate(nxyz_list):
            mol_index = mol_indices[i]

            sub_dic[mol_index]['nxyz'].append(nxyz)
            sub_dic[mol_index]['energies'].append(energy_list[i])


def update_conf_dic_info_list(info_list,
                              conf_dic,
                              sampled_ens,
                              stage,
                              overwrite):

    sub_dic = conf_dic[stage]

    # over-write any old nxyzs, so we can use all nxyzs in this dictionary
    # to make atomsbatches for optimization

    if overwrite:
        for key in sub_dic.keys():
            sub_dic[key]['nxyz'] = []
            sub_dic[key]['energies'] = []

    for info, these_ens in zip(info_list, sampled_ens):
        mol_indices = [info["mol_index"]] * len(these_ens)
        nxyz_list = info['nxyz_list']

        for i, nxyz in enumerate(nxyz_list):
            mol_index = mol_indices[i]

            sub_dic[mol_index]['nxyz'].append(nxyz)
            sub_dic[mol_index]['energies'].append(these_ens[i])


def sampled_to_atoms_batches(conf_dic,
                             key,
                             all_params,
                             num_parallel,
                             model):

    sub_dic = conf_dic[key]

    new_params_list = []
    mol_indices = []

    for mol_index, dic in sub_dic.items():

        nxyz_list = dic['nxyz']

        for nxyz in nxyz_list:
            rep_params = [params for params in all_params if
                          params['mol_index'] == mol_index][0]
            these_params = copy.deepcopy(rep_params)
            these_params['nxyz'] = nxyz
            new_params_list.append(these_params)
            mol_indices.append(mol_index)

    # batch
    num_split = int(np.ceil(len(new_params_list) / num_parallel))
    batched_params_list = np.split(new_params_list, [num_parallel * i for
                                                     i in range(1, num_split)])
    split_mol_idx = np.split(mol_indices, [num_parallel * i for
                                           i in range(1, num_split)])

    atoms_batches = []
    for this_params_list in batched_params_list:
        atoms_batch = params_to_atoms_batch(params_list=this_params_list,
                                            model=model)
        atoms_batches.append(atoms_batch)

    return atoms_batches, split_mol_idx, batched_params_list


def trim_sampled(nxyz_lists,
                 this_params_list,
                 sampled_ens):

    info_list = []
    new_sampled_ens = []

    for i, these_params in enumerate(this_params_list):

        these_params["steps"] = int(np.ceil(these_params["mtd_time"] * 1000 /
                                            these_params["time_step"]))
        num_poses = get_num_starting_poses(params=these_params,
                                           md_type="MD")
        nxyz_list = [lst[i] for lst in nxyz_lists[:num_poses]]
        these_sampled_ens = [lst[i] for lst in sampled_ens[:num_poses]]

        info_list.append({"nxyz_list": nxyz_list,
                          "mol_index": these_params["mol_index"]})
        new_sampled_ens.append(these_sampled_ens)

    return info_list, new_sampled_ens


def sample_confs(base_dir,
                 this_params_list,
                 model,
                 conf_dic,
                 all_params,
                 num_parallel,
                 params):

    trj_filename = get_trj_file(params)
    num_poses = get_num_starting_poses(params=params,
                                       md_type="MD")

    path = os.path.join(base_dir, trj_filename)
    trj = Trajectory(path)

    # don't use first frame - that was already tightly optimized
    # from the previous round, whereas the others are only loosely optimized,
    # so we don't want that to throw off the thresholding

    all_steps = np.arange(len(trj))[1:]
    denom = max([(num_poses - 1), 1])
    num_skip = int(np.floor((len(all_steps) - 1) /
                            denom))
    sampled_idx = all_steps[::num_skip][:num_poses]

    sampled_confs = [trj[idx] for idx in sampled_idx]
    sampled_ens = [conf.get_potential_energy().tolist() for
                   conf in sampled_confs]

    # update the master `conf_dic` dictionary

    num_atoms = np.array([len(these_params['nxyz'])
                          for these_params in this_params_list])
    nxyz_lists = [atoms_batch_to_nxyz(conf, num_atoms) for conf in
                  sampled_confs]

    # the actual number of starting poses needed for each trajectory will be <= num_poses,
    # because we used the maximum MTD time among all molecules. So for any molecule
    # with mtd_time < max_mtd_time, we need fewer samples. This saves time during opt
    # and makes the final results independent of the batch the molecules are in, but it
    # also means we waste some MTD time

    info_list, sampled_ens = trim_sampled(nxyz_lists=nxyz_lists,
                                          this_params_list=this_params_list,
                                          sampled_ens=sampled_ens)

    update_conf_dic_info_list(info_list=info_list,
                              conf_dic=conf_dic,
                              sampled_ens=sampled_ens,
                              stage='sampled',
                              overwrite=True)


def trim(conf_dic,
         stage,
         window):

    sub_dic = conf_dic[stage]

    for mol_index, dic in sub_dic.items():
        these_ens = np.array(dic['energies'])
        if these_ens.shape[0] == 0:
            continue
        rel_ens = (these_ens - np.min(these_ens)) * const.EV_TO_KCAL_MOL
        valid_idx = (rel_ens <= window).nonzero()[0]

        vowel_ints = [8, 11, 18]
        suffix = "n" if any([str(window).startswith(str(i))
                             for i in vowel_ints]) else ""

        print("%d conformers from molecule %d remain within a%s %.2f kcal/mol window" % (
            len(valid_idx), mol_index + 1, suffix, window))

        dic['energies'] = [dic['energies'][i] for i in valid_idx]
        dic['nxyz'] = [dic['nxyz'][i] for i in valid_idx]


def get_fmax(forces):
    return ((forces ** 2).sum(axis=1) ** 0.5).max()


def update_conf_dic_w_nxyz(conf_dic,
                           atoms_batches,
                           mol_indices,
                           stage,
                           overwrite,
                           fmax,
                           only_converged):

    ens = []
    forces = []
    nxyz_list = []

    for atoms_batch in atoms_batches:
        ens.append(atoms_batch.get_potential_energy())
        forces += np.split(atoms_batch.get_forces(),
                           np.cumsum(atoms_batch.num_atoms))[:-1]
        nxyz_list += atoms_batch_to_nxyz(atoms_batch)

    ens = np.concatenate(ens)
    cat_mol_idx = np.concatenate(mol_indices)

    sub_dic = conf_dic[stage]

    for mol_index in range(cat_mol_idx.max() + 1):
        batch_idx = (cat_mol_idx == mol_index).nonzero()[0]
        # remove any nans or infinities
        batch_idx = (batch_idx[np.isfinite(ens[batch_idx])])

        # remove anything not converged, e.g. from multi-level opt
        # where the geometry wasn't converged the first time round

        these_forces = [forces[i] for i in batch_idx]

        if only_converged:
            convgd = np.array([get_fmax(f) < fmax for
                               f in these_forces]).astype('bool')
            batch_idx = batch_idx[convgd]

        # get the associated energies and positions
        these_ens = ens[batch_idx].tolist()
        this_nxyz = [nxyz_list[i] for i in batch_idx]

        update_dic = {"nxyz": this_nxyz,
                      "energies": these_ens}

        for key, val in update_dic.items():
            if overwrite:
                sub_dic[mol_index][key] = val
            else:
                sub_dic[mol_index][key] += val


def opt_files_to_common_dic(all_opt_files):
    file_dic = {}
    for file in all_opt_files:
        if file is None:
            continue
        split = file.split("/")
        job_dir = "/".join(split[:-1])
        name = split[-1]

        if job_dir not in file_dic:
            file_dic[job_dir] = []
        file_dic[job_dir].append(name)

    return file_dic


def combine_separate_opts(all_opt_files,
                          conf_dic,
                          params_list,
                          stage):
    """
    We save every conformer as its own separate opt file in each sub-directory, and
    we want to combine the files with conformers from the same optimization round.
    This function combines all files from a given optimization round, as given in
    `all_opt_files`.
    """

    file_dic = opt_files_to_common_dic(all_opt_files)
    save_paths = []

    for job_dir, file_names in file_dic.items():
        trj = []
        paths = []

        for file_name in file_names:
            path = os.path.join(job_dir, file_name)
            trj += [i for i in Trajectory(path)]
            paths.append(path)
            os.remove(path)

        save_path = sort_trj_files(paths)[0]

        trj_writer = TrajectoryWriter(filename=save_path)
        for atoms in trj:
            trj_writer.write(atoms)
        trj_writer.close()

        save_paths.append(save_path)

    return save_paths


def opt_from_conf_dic(params,
                      model,
                      conf_dic,
                      all_params,
                      num_parallel,
                      prev_stage,
                      current_stage,
                      fmax_key,
                      window_key,
                      overwrite,
                      base_dir,
                      params_list,
                      only_converged):

    out = sampled_to_atoms_batches(conf_dic=conf_dic,
                                   key=prev_stage,
                                   all_params=all_params,
                                   num_parallel=num_parallel,
                                   model=model)
    atoms_batches, mol_indices, batched_params_list = out

    opt_params = copy.deepcopy(params)
    fmax = params[fmax_key]
    opt_params.update({"fmax": fmax})

    print("Optimizing %d conformers..." % (len(atoms_batches)))

    opt_atoms_batches = []
    all_opt_files = []

    for i, this_params_list in enumerate(batched_params_list):
        atoms_batch = atoms_batches[i]
        these_opt, _ = confs_to_opt(params=opt_params,
                                    best_confs=[atoms_batch],
                                    model=model,
                                    return_sorted=False,
                                    ref_idx=i)
        opt_atoms_batches += these_opt

        all_opt_files += save_separate_opts(base_dir=base_dir,
                                            this_params_list=this_params_list)

    update_conf_dic_w_nxyz(conf_dic=conf_dic,
                           atoms_batches=opt_atoms_batches,
                           mol_indices=mol_indices,
                           stage=current_stage,
                           overwrite=overwrite,
                           fmax=fmax,
                           only_converged=only_converged)

    trim(conf_dic=conf_dic,
         stage=current_stage,
         window=params[window_key])

    print("Removing duplicates among conformers optimized with %s thresholds..."
          % current_stage)

    batched_dedupe(conf_dic=conf_dic,
                   stage=current_stage,
                   params=params)

    save_paths = combine_separate_opts(all_opt_files=all_opt_files,
                                       conf_dic=conf_dic,
                                       params_list=params_list,
                                       stage=current_stage)

    return save_paths


def load_unconverged(opt_params,
                     all_opt_files,
                     all_params,
                     conf_dic,
                     prev_stage):

    fmax_thresh = opt_params["fmax"]
    opt_file_dic = opt_files_to_common_dic(all_opt_files)

    for job_dir, opt_paths in opt_file_dic.items():
        mol_index = None
        for file_name in opt_paths:

            path = os.path.join(job_dir, file_name)
            trj = Trajectory(path)
            unconverged_idx = [i for i, atoms in enumerate(trj) if
                               np.sqrt((atoms.get_forces() ** 2)
                                       .sum(axis=1).max()) > fmax_thresh]

            if mol_index is None:
                mol_index = [i for i in all_params if i['job_dir'] == job_dir
                             ][0]['mol_index']
                old_sub_dic = conf_dic[prev_stage][mol_index]
                old_sub_dic['nxyz'] = []
                old_sub_dic['energies'] = []

            nxyz_list = []
            en_list = []

            for i in unconverged_idx:
                atoms = trj[i]
                nxyz = np.concatenate([atoms.get_atomic_numbers().reshape(-1, 1),
                                       atoms.get_positions()],
                                      axis=-1)
                nxyz_list.append(nxyz)
                en_list.append(atoms.get_potential_energy().item())

            old_sub_dic['nxyz'] += nxyz_list
            old_sub_dic['energies'] += en_list


def multi_step_batched_opt(batched_params_list,
                           atoms_batches,
                           model,
                           opt_params,
                           base_dir,
                           prev_stage,
                           current_stage,
                           all_params,
                           conf_dic,
                           mol_indices,
                           num_parallel):

    opt_atoms_batches = []
    all_opt_files = []
    new_mol_indices = []

    iterative_max_steps = INIT_MAX_ITER[current_stage]
    max_i = len(iterative_max_steps)

    for i in range(max_i + 1):

        new_mol_indices += mol_indices

        use_params = copy.deepcopy(opt_params)
        if i == max_i:
            opt_max_step = (max([opt_params.get("opt_max_step", 1500),
                                 iterative_max_steps[-1] + 1]) -
                            iterative_max_steps[-1])
        else:

            opt_max_step = copy.deepcopy(iterative_max_steps[i])

            # Because you actually want the delta in number of steps. E.g.
            # 30 -> 45 means you already did 30 steps, so in your next round you just
            # want to do an additional 15

            if i != 0:
                opt_max_step -= iterative_max_steps[i - 1]

        use_params['opt_max_step'] = opt_max_step

        num_confs = len(batched_params_list)
        suffix = "" if num_confs == 1 else "s"
        final = "" if num_confs == 1 else " each"
        print("Sub-round %d: Optimizing %d conformer%s for %d steps%s..." % (
            i + 1, num_confs, suffix, opt_max_step, final))

        recent_opt_files = []
        for j, this_params_list in enumerate(batched_params_list):
            atoms_batch = atoms_batches[j]
            these_opt, _ = confs_to_opt(params=use_params,
                                        best_confs=[atoms_batch],
                                        model=model,
                                        return_sorted=False,
                                        ref_idx=j)
            opt_atoms_batches += these_opt

            recent_opt_files += save_separate_opts(base_dir=base_dir,
                                                   this_params_list=this_params_list)

        all_opt_files += recent_opt_files

        if i == max_i:
            break

        load_unconverged(opt_params=opt_params,
                         all_opt_files=recent_opt_files,
                         all_params=all_params,
                         conf_dic=conf_dic,
                         prev_stage=prev_stage)

        if not any([val['nxyz'] for val in conf_dic[prev_stage].values()]):
            break

        out = sampled_to_atoms_batches(conf_dic=conf_dic,
                                       key=prev_stage,
                                       all_params=all_params,
                                       num_parallel=num_parallel,
                                       model=model)
        atoms_batches, mol_indices, batched_params_list = out

    return opt_atoms_batches, all_opt_files, new_mol_indices


def multi_step_opt_from_conf_dic(params,
                                 model,
                                 conf_dic,
                                 all_params,
                                 num_parallel,
                                 prev_stage,
                                 current_stage,
                                 fmax_key,
                                 window_key,
                                 overwrite,
                                 base_dir,
                                 params_list,
                                 only_converged):

    out = sampled_to_atoms_batches(conf_dic=conf_dic,
                                   key=prev_stage,
                                   all_params=all_params,
                                   num_parallel=num_parallel,
                                   model=model)
    atoms_batches, mol_indices, batched_params_list = out

    opt_params = copy.deepcopy(params)
    fmax = params[fmax_key]
    opt_params.update({"fmax": fmax})

    print("Optimizing %d conformers..." % (len(atoms_batches)))

    out = multi_step_batched_opt(batched_params_list=batched_params_list,
                                 atoms_batches=atoms_batches,
                                 model=model,
                                 opt_params=opt_params,
                                 base_dir=base_dir,
                                 prev_stage=prev_stage,
                                 current_stage=current_stage,
                                 all_params=all_params,
                                 conf_dic=conf_dic,
                                 mol_indices=mol_indices,
                                 num_parallel=num_parallel)

    opt_atoms_batches, all_opt_files, mol_indices = out

    update_conf_dic_w_nxyz(conf_dic=conf_dic,
                           atoms_batches=opt_atoms_batches,
                           mol_indices=mol_indices,
                           stage=current_stage,
                           overwrite=overwrite,
                           fmax=fmax,
                           only_converged=only_converged)

    trim(conf_dic=conf_dic,
         stage=current_stage,
         window=params[window_key])

    print("Removing duplicates among conformers optimized with %s thresholds..."
          % current_stage)

    batched_dedupe(conf_dic=conf_dic,
                   stage=current_stage,
                   params=params)

    save_paths = combine_separate_opts(all_opt_files=all_opt_files,
                                       conf_dic=conf_dic,
                                       params_list=params_list,
                                       stage=current_stage)

    return save_paths


def coarse_opt(params,
               model,
               conf_dic,
               all_params,
               num_parallel,
               base_dir,
               params_list):

    func = multi_step_opt_from_conf_dic
    # func = opt_from_conf_dic

    out = func(params=params,
               model=model,
               conf_dic=conf_dic,
               all_params=all_params,
               num_parallel=num_parallel,
               prev_stage='sampled',
               current_stage='coarse',
               fmax_key='fmax_coarse',
               window_key='window_coarse',
               overwrite=True,
               base_dir=base_dir,
               params_list=params_list,
               only_converged=True)
    return out


def tight_opt(params,
              model,
              conf_dic,
              all_params,
              num_parallel,
              base_dir,
              params_list):

    func = multi_step_opt_from_conf_dic
    # func = opt_from_conf_dic

    out = func(params=params,
               model=model,
               conf_dic=conf_dic,
               all_params=all_params,
               num_parallel=num_parallel,
               prev_stage='coarse',
               current_stage='tight',
               fmax_key='fmax_tight',
               window_key='window_tight',
               overwrite=False,
               base_dir=base_dir,
               params_list=params_list,
               only_converged=True)
    return out


def vtight_opt(params,
               model,
               conf_dic,
               all_params,
               num_parallel,
               base_dir,
               params_list):

    func = multi_step_opt_from_conf_dic
    # func = opt_from_conf_dic

    out = func(params=params,
               model=model,
               conf_dic=conf_dic,
               all_params=all_params,
               num_parallel=num_parallel,
               prev_stage='tight',
               current_stage='vtight',
               fmax_key='fmax_vtight',
               window_key='window_vtight',
               overwrite=False,
               base_dir=base_dir,
               params_list=params_list,
               only_converged=False)

    return out


def num_to_symbols(numbers):
    symbols = [PERIODICTABLE.GetElementSymbol(int(number))
               for number in numbers]

    return symbols


def write_confs_xyz(nxyz_list,
                    energy_list,
                    path):

    au_energies = np.array([float(energy) * const.EV_TO_AU for
                            energy in energy_list])

    # Crazy weird bug where an energy of 0 means the conformer gets removed
    # In that case, add a constant shift, which shouldn't affect any of the results

    if (abs(au_energies) < 1e-3).any():
        au_energies += (-au_energies.min() + 0.5)

    text = ""
    for i, nxyz in enumerate(nxyz_list):
        energy = au_energies[i]

        comment = "%.8f !CONF%d" % (energy, i + 1)

        this_text = make_xyz_text(positions=nxyz[:, 1:],
                                  symbols=num_to_symbols(nxyz[:, 0]),
                                  comment=comment)

        if i != 0:
            text += "\n"
        text += this_text

    with open(path, 'w') as f:
        f.write(text)


def run_cre_check(nxyz_list,
                  energy_list,
                  ethr,
                  rthr,
                  bthr,
                  ewin,
                  mol_index):
    """
    Same idea as `run_cre_check` in `neural_confgen`, but using atomic numbers, positions
    and energies instead of Atoms objects. This is so that we don't need to make whole new
    atoms objects for our batches, with properties stored in their calculators. And
    can instead just slice their positions and energies into batches
    """

    crest_path = os.path.join(os.environ["CONDA_PREFIX"], 'bin/crest')
    base_name = make_rand_string()
    job_dir = os.path.join("/tmp", base_name)

    cwd = os.getcwd()
    os.makedirs(job_dir)
    os.chdir(job_dir)

    confs_path = os.path.join(job_dir, "confs.xyz")
    conf_0_path = os.path.join(job_dir, "conf_0.xyz")

    # first sort by energy

    argsort = np.argsort(energy_list)
    nxyz_list = [nxyz_list[i] for i in argsort]
    energy_list = [energy_list[i] for i in argsort]

    write_confs_xyz(nxyz_list=nxyz_list,
                    energy_list=energy_list,
                    path=confs_path)
    write_confs_xyz(nxyz_list=nxyz_list[:1],
                    energy_list=energy_list[:1],
                    path=conf_0_path)

    command = ("%s %s -cregen %s -ethr %.6f -rthr %.6f -bthr %.6f -ewin %.6f -enso "
               "> cregen.out" % (crest_path, conf_0_path, confs_path, ethr, rthr,
                                 bthr, ewin))

    p = bash_command(command)
    p.wait()

    unique_idx = read_unique(job_dir=job_dir)

    if unique_idx is None:
        unique_idx = list(range(len(nxyz_list)))

    # sort by energy
    unique_idx = sorted(unique_idx, key=lambda x: energy_list[x])
    num_removed = len(nxyz_list) - len(unique_idx)

    plural = 's' if num_removed > 1 else ''
    print("Removed %d duplicate conformer%s from molecule %d with cregen" % (
        num_removed, plural, mol_index + 1))
    sys.stdout.flush()

    nxyz_list = [nxyz_list[i] for i in unique_idx]
    energy_list = [energy_list[i] for i in unique_idx]

    shutil.rmtree(job_dir)
    os.chdir(cwd)

    return nxyz_list, energy_list, unique_idx


def cre_wrapper(kwargs):
    return run_cre_check(**kwargs)


def run_batched_cre_check(conf_dic,
                          stage,
                          **kwargs):

    nxyz_lists = []
    energy_lists = []
    all_mol_idx = []

    dic = conf_dic[stage]
    for mol_idx, sub_dic in dic.items():
        lsts = []

        for key in ['nxyz', 'energies']:
            lsts.append(sub_dic[key])

        nxyz_list, energy_list = lsts

        if not nxyz_list:
            continue

        nxyz_lists.append(nxyz_list)
        energy_lists.append(energy_list)
        all_mol_idx.append(mol_idx)

    kwargs_list = [dict(nxyz_list=n,
                        energy_list=energy_lists[i],
                        mol_index=all_mol_idx[i],
                        **kwargs) for i, n in
                   enumerate(nxyz_lists)]

    # don't do this in parallel - that can give a small speedup, but it causes all
    # sorts of problems when you have lots of molecules, probably because of some
    # issue with how crest allocates threads / cores. Just do it in series

    # pool = mp.Pool(mp.cpu_count())
    # pairs = pool.map(cre_wrapper, kwargs_list)
    # pool.close()

    pairs = [cre_wrapper(these_kwargs) for these_kwargs in kwargs_list]

    # trim
    nxyz_lists = [pair[0] for pair in pairs]
    energy_lists = [pair[1] for pair in pairs]
    unique_idx_list = [pair[2] for pair in pairs]

    # update conf_dic

    for i, nxyz_list in enumerate(nxyz_lists):
        energy_list = energy_lists[i]
        mol_idx = all_mol_idx[i]
        unique_idx = unique_idx_list[i]

        sub_dic = conf_dic[stage][mol_idx]
        sub_dic.update({'nxyz': nxyz_list,
                        'energies': energy_list,
                        'unique_idx': unique_idx})


def batched_dedupe(conf_dic,
                   stage,
                   params):
    dedupe_params = params["crest_dedupe"]
    if not dedupe_params["on"]:
        print("De-duplication turned off by user request.")
        return

    crest_params = dedupe_params["params"]
    run_batched_cre_check(conf_dic=conf_dic,
                          stage=stage,
                          ethr=crest_params["ethr"],
                          rthr=crest_params["rthr"],
                          bthr=crest_params["bthr"],
                          ewin=crest_params["ewin"])


def init_conf_dic(num_mols):
    """
    Initialize a dictionary that holds all the coarsely- and tightly-optimized conformers
    from each different molecule, plus the initial sampled molecules from the MTD run.
    The dictionary has the following form:

    {"sampled": {0: {"nxyz": [], "atoms_batches": [], "energies": []},
    1: {"nxyz": [], "atoms_batches": [], "energies": []}, ...,
    <num_molecules - 1 >: {"nxyz": [], "atoms_batches": [], "energies": []}
    },


    "coarse": {0: {"nxyz": [], "atoms_batches": [], "energies": []},
    1: {"nxyz": [], "atoms_batches": [], "energies": []}, ...,
    <num_molecules - 1>: {"nxyz": [], "atoms_batches": [], "energies": []}
    },

     "tight": {0: {"nxyz": [], "atoms_batches": [], "energies": []},
     1: {"nxyz": [], "atoms_batches": [], "energies": []}, ...,
     <num_molecules - 1>: {"nxyz": [], "atoms_batches": [], "energies": []}
     }
    }


    Note that the idea is to re-set `nxyz` and `atoms_batches` to an empty list
    for both "sampled" and "coarse" before each new round. Then we update them with
    the geometries from this round only. That way when we're doing the subsequent
    optimization, we can just take all the geometries from the proper key and
    mol index and optimize all of them. On the other hand, "tight" accumulates all
    geometries that made it past "coarse" and de-duplication.
    """

    keys = ["coarse", "tight", "vtight", "sampled"]
    conf_dic = {key: {} for key in keys}

    for key in keys:
        sub_dic = conf_dic[key]
        for i in range(num_mols):
            sub_dic.update({i: {"nxyz": [],
                                # "atoms_batches": [],
                                "energies": []}
                            })

    progress_keys = ["retry_attempts", "num_md_runs"]
    for key in progress_keys:
        conf_dic[key] = {i: 0 for i in range(num_mols)}

    return conf_dic


def batched_is_done(conf_dic,
                    old_en_dic,
                    params):

    is_done_list = []
    for mol_index, old_ens in old_en_dic.items():
        if not old_ens:
            continue

        # update with the number of MD runs this molecule has been though
        conf_dic["num_md_runs"][mol_index] += 1

        print("Molecule %d: " % (mol_index + 1))
        new_ens = conf_dic['tight'][mol_index]['energies']
        is_done = get_is_done(old_ens=old_ens,
                              new_ens=new_ens,
                              params=params)

        # mark as done if it's exceeded `max_md_runs`, and not done if it hasn't
        # exceeded `min_md_runs`
        num_md_runs = conf_dic["num_md_runs"][mol_index]

        min_md_runs = params["min_md_runs"]
        max_md_runs = params["max_md_runs"]

        if num_md_runs < min_md_runs:
            suffix = "s" if (min_md_runs > 1) else ""
            print("Not yet reached the minimum of %d MD run%s for molecule %d. "
                  "Continuing." % (min_md_runs, suffix, (mol_index + 1)))
            is_done = False

        if num_md_runs >= max_md_runs:
            suffix = "s" if (max_md_runs > 1) else ""
            print("Reached the maximum of %d MD run%s for molecule %d. Breaking now. "
                  % (max_md_runs, suffix, (mol_index + 1)))
            is_done = True

        if is_done:
            is_done_list.append(mol_index)

    return is_done_list


def initial_tight_opt(model,
                      conf_dic,
                      params_list,
                      num_parallel,
                      base_dir):

    print("Optimizing input conformers with tight thresholds...")

    num_split = int(np.ceil(len(params_list) / num_parallel))
    split_params = np.split(params_list, [num_parallel * i for
                                          i in range(1, num_split)])

    for this_params_list in split_params:

        atoms_batch = params_to_atoms_batch(this_params_list,
                                            model)

        opt_params = copy.deepcopy(this_params_list[0])
        opt_params.update({"fmax": opt_params["fmax_tight"]})

        opt_atoms_batches, _ = confs_to_opt(params=opt_params,
                                            best_confs=[atoms_batch],
                                            model=model,
                                            return_sorted=False)

        opt_atoms_batch = opt_atoms_batches[0]
        save_separate_opts(base_dir=base_dir,
                           this_params_list=this_params_list)

        nxyz_list = atoms_batch_to_nxyz(opt_atoms_batch)
        sampled_ens = opt_atoms_batch.get_potential_energy().tolist()

        update_conf_dic_using_params(nxyz_lists=[nxyz_list],
                                     this_params_list=this_params_list,
                                     conf_dic=conf_dic,
                                     sampled_ens=[sampled_ens],
                                     stage='tight',
                                     overwrite=False)

    print("Initial optimization complete!")


def make_params_list(num_parallel,
                     params_list,
                     is_done_list,
                     conf_dic):

    mol_idx = list(range(len(params_list)))
    use_idx = [i for i in mol_idx if (i not in is_done_list) and
               (conf_dic["retry_attempts"][i] <= MAX_RETRIES)][:num_parallel]
    this_params_list = []

    for idx in use_idx:
        these_params = copy.deepcopy(params_list[idx])
        sub_dic = conf_dic['tight'][idx]

        arg = np.argsort(sub_dic['energies'])[0]
        nxyz = sub_dic['nxyz'][arg]

        these_params['nxyz'] = nxyz
        this_params_list.append(these_params)

    return this_params_list


def init_round(conf_dic,
               params_list,
               num_parallel,
               is_done_list,
               model):

    this_params_list = make_params_list(num_parallel=num_parallel,
                                        params_list=params_list,
                                        is_done_list=is_done_list,
                                        conf_dic=conf_dic)

    atoms_batch = params_to_atoms_batch(params_list=this_params_list,
                                        model=model)
    params = get_representative_params(this_params_list)

    return atoms_batch, this_params_list, params


def sort_trj_files(paths):
    sort_files = sorted(paths, key=lambda x: int(x.split(".traj")[0]
                                                 .split("_")[-1]))
    return sort_files


def trj_to_match_nxyz(path,
                      conf_dic,
                      stage,
                      mol_index):

    trj = Trajectory(path)
    trj_xyz = np.stack([i.get_positions() for i in trj])
    nxyz_list = conf_dic[stage][mol_index]['nxyz']

    idx = []

    for i, nxyz in enumerate(nxyz_list):
        xyz = np.array(nxyz)[:, 1:]
        distances = (((xyz - trj_xyz) ** 2).sum((-1, -2)) /
                     xyz.shape[0]) ** 0.5
        closest_idx = np.argsort(distances)[0]
        if distances[int(closest_idx)] > 1e-3:
            continue

        idx.append(int(closest_idx))

    trj = [trj[i] for i in idx]
    return trj


def get_unique_params(all_params):
    param_dic = {}
    for params in all_params:
        mol_index = params['mol_index']
        if mol_index not in param_dic:
            param_dic[mol_index] = params
    unique_params = list(param_dic.values())

    return unique_params


def save_final_confs(conf_dic,
                     all_params):

    # for use in `nn_refine_crest`
    unique_params = get_unique_params(all_params)

    for params in unique_params:
        mol_index = params["mol_index"]
        retry_attempts = conf_dic["retry_attempts"][mol_index]

        # this means it never succeeded, so the last opt file is not actually
        # the final vtight optimization. By making sure that file doesn't exist,
        # we also ensure that the parser makes an error when parsing it, so that
        # these results don't make it into the DB

        if retry_attempts > MAX_RETRIES:
            continue

        job_dir = params["job_dir"]
        opt_paths = [os.path.join(job_dir, i) for i in
                     os.listdir(job_dir) if 'opt' in i and i.endswith('traj')]

        final_opt_path = sort_trj_files(opt_paths)[-1]

        # sort and filter by energy and write trajectory
        # can't just apply `unique_idx` to the saved trajectory here, so need
        # to get the forces by comparing nxyz to trjs

        trj = trj_to_match_nxyz(path=final_opt_path,
                                conf_dic=conf_dic,
                                stage='vtight',
                                mol_index=mol_index)
        ens = np.concatenate([atoms.get_potential_energy() for atoms in trj])
        rel_ens = (ens - np.min(ens)) * const.EV_TO_KCAL_MOL

        args = np.argsort(ens)
        idx = args[rel_ens[args] <= params["window_vtight"]]
        trj = [trj[i] for i in idx]

        # save to new path
        new_path = os.path.join(job_dir, FINAL_OPT_FILENAME)
        writer = TrajectoryWriter(new_path)
        for atoms in trj:
            writer.write(atoms)
        writer.close()


def clean_and_summarize(all_params):

    # for use in `nn_refine_crest`
    unique_params = get_unique_params(all_params)

    for params in unique_params:
        job_dir = params['job_dir']
        opt_path = os.path.join(job_dir, FINAL_OPT_FILENAME)
        if not os.path.isfile(opt_path):
            continue
        confs = Trajectory(opt_path)
        print("For molecule %d: " % (params['mol_index'] + 1))
        summarize_final(confs=confs,
                        params=params)
        print()

        # combine all its `atoms.traj` into one, and same for `opt.traj`

        for name in ['atoms', 'opt']:
            files = [os.path.join(job_dir, i) for i in os.listdir(job_dir)
                     if i.startswith(name) and i.endswith(".traj")]
            files = sort_trj_files([i for i in files if os.path.isfile(i)])
            trj = []
            for file in files:
                trj += list(Trajectory(file))
                os.remove(file)

            new_file = os.path.join(job_dir, '%s_0.traj' % name)
            writer = TrajectoryWriter(new_file)
            for atoms in trj:
                writer.write(atoms)
            writer.close()


def get_starting_data(base_dir):
    direc = os.path.dirname(os.path.abspath(__file__))
    args = parse_args(arg_path=ARGS_PATH,
                      direc=direc)
    num_parallel = args.np

    params_list = load_all_params(base_dir=base_dir,
                                  args=args)
    params = params_list[0]
    model = get_model(params)

    conf_dic = init_conf_dic(num_mols=len(params_list))

    return params_list, model, num_parallel, conf_dic


def update_w_error(this_params_list,
                   conf_dic,
                   error,
                   is_done_list):

    err_str = str(error)
    bad_mol_idx = None

    # if this is part of the error message then we know which molecule
    # had the problem

    if "is zero, singular U" in err_str:
        # example: torch.linalg.eigh: For batch 4: U(3,3) is zero, singular U.
        try:
            bad_idx = int(err_str.split("batch")[1].split(":")[0])
            bad_mol_idx = [this_params_list[bad_idx]["mol_index"]]
            print(("Failure was specifically for molecule %d " %
                   (bad_mol_idx[0] + 1)))

        except ValueError:
            pass
    elif "the input matrix is ill-conditioned" in err_str:
        # example: Failed with error 'torch.linalg.eigh: (Batch element 5): The
        # algorithm failed to converge because the input matrix is ill-conditioned or
        # has too many repeated eigenvalues (error code: 3).'

        try:
            bad_idx = int(err_str.split("Batch element")[1].split(")")[0])
            bad_mol_idx = [this_params_list[bad_idx]["mol_index"]]
            print(("Failure was specifically for molecule %d " %
                   (bad_mol_idx[0] + 1)))
        except ValueError:
            pass

    # otherwise just give each molecule the error
    if bad_mol_idx is None:
        print(("Couldn't figure out which molecule failed. Counting this as a "
               "failed attempt for all molecules in the batch"))
        bad_mol_idx = [params['mol_index'] for params in this_params_list]

    for mol_idx in bad_mol_idx:
        retry_attempts = conf_dic["retry_attempts"][mol_idx]
        conf_dic["retry_attempts"][mol_idx] += 1

        if (retry_attempts + 1) > MAX_RETRIES:
            string = "retry" if MAX_RETRIES == 1 else "retries"
            print("Exceeded %d %s for molecule %d. Not retrying it again. "
                  % (retry_attempts, string, mol_idx + 1))

            is_done_list.append(mol_idx)
            is_done_list = list(set(is_done_list))

    return is_done_list


def get_sub_dirs(base_dir):
    sub_dirs = [os.path.join(base_dir, i) for i in os.listdir(base_dir)]
    sub_dirs = [i for i in sub_dirs if os.path.isdir(i)]

    return sub_dirs


def get_approx_times(base_dir,
                     total_time,
                     num_parallel):
    """
    Approximate the time taken for each individual sub-job, based on how long
    the total job took, and how many rounds the sub-job went through. The purpose is
    to give a reasonable estimate of job duration so we can store it in the DB.
    """

    # dictionary that tells you how many rounds each sub-job went through
    rounds = {}
    sub_dirs = get_sub_dirs(base_dir)

    for sub_dir in sub_dirs:
        mtd_files = [i for i in os.listdir(sub_dir) if i.startswith("atoms_") and
                     i.endswith(".traj")]
        rounds[sub_dir] = len(mtd_files)

        total_rounds = np.max(list(rounds.values()))
        num_mols = len(rounds)

    approx_times = {}

    eff_num_parallel = np.min([num_parallel, num_mols])
    for sub_dir, num_rounds in rounds.items():
        if total_rounds == 0:
            approx_times[sub_dir] = total_time
        else:
            approx_times[sub_dir] = (total_time * (num_rounds / total_rounds) *
                                     (eff_num_parallel / num_mols))

    return approx_times


def make_completion_files(base_dir,
                          num_parallel,
                          start,
                          end):
    """
    Write log files for each sub-job so that it can be properly parsed into the database.
    Each log file needs only a line with the total time taken by that job. This line
    shows that the job is done, and also gives a duration that can be parsed into the
    database.
    """

    total_time = (end - start)
    approx_times = get_approx_times(base_dir=base_dir,
                                    total_time=total_time,
                                    num_parallel=num_parallel)

    sub_dirs = get_sub_dirs(base_dir)
    for sub_dir in sub_dirs:
        # If it doesn't have the final opt traj file then it didn't succeed
        opt_file = os.path.join(sub_dir, FINAL_OPT_FILENAME)
        if not os.path.isfile(opt_file):
            continue

        this_time = approx_times[sub_dir]
        msg = time_to_msg(seconds=this_time)
        log_file = os.path.join(sub_dir, LOG_FILE)
        with open(log_file, "w") as f:
            f.write(msg)


def get_old_ens(this_params_list,
                conf_dic):
    current_mol_idx = [these_params['mol_index'] for these_params
                       in this_params_list]
    old_en_dic = {mol_idx: copy.deepcopy(val['energies'])
                  for mol_idx, val in conf_dic['tight'].items()
                  if mol_idx in current_mol_idx}

    return old_en_dic


def update_is_done(conf_dic,
                   old_en_dic,
                   params,
                   is_done_list):

    is_done_list += batched_is_done(conf_dic=conf_dic,
                                    old_en_dic=old_en_dic,
                                    params=params)
    is_done_list = list(set(is_done_list))

    return is_done_list


def init_is_done_list(params_list):
    is_done_list = []
    for params in params_list:
        max_md_runs = params["max_md_runs"]
        if max_md_runs == 0:
            is_done_list.append(params['mol_index'])
    return is_done_list


def set_mtd_times(base_dir):
    for i in os.listdir(base_dir):
        path = os.path.join(base_dir, i, 'job_info.json')
        if not os.path.isfile(path):
            continue

        with open(path, 'r') as f:
            params = json.load(f)
        set_mtd_time(info_file=path,
                     params=params)


def main(base_dir):

    start = time.time()

    params_list, model, num_parallel, conf_dic = get_starting_data(base_dir)
    initial_tight_opt(model=model,
                      conf_dic=conf_dic,
                      params_list=params_list,
                      num_parallel=num_parallel,
                      base_dir=base_dir)

    is_done_list = init_is_done_list(params_list=params_list)
    cwd = os.getcwd()

    while len(is_done_list) < len(params_list):

        atoms_batch, this_params_list, params = init_round(conf_dic=conf_dic,
                                                           params_list=params_list,
                                                           num_parallel=num_parallel,
                                                           is_done_list=is_done_list,
                                                           model=model)

        # to make sure we don't get stuck in the wrong directory in the middle
        os.chdir(cwd)

        try:
            run_md(atoms_batch=atoms_batch,
                   this_params_list=this_params_list,
                   base_dir=base_dir,
                   params=params)

            sample_confs(base_dir=base_dir,
                         this_params_list=this_params_list,
                         model=model,
                         conf_dic=conf_dic,
                         all_params=params_list,
                         num_parallel=num_parallel,
                         params=params)

            print("Optimizing with coarse thresholds...")
            coarse_opt(params=params,
                       model=model,
                       conf_dic=conf_dic,
                       all_params=params_list,
                       num_parallel=num_parallel,
                       base_dir=base_dir,
                       params_list=params_list)

            old_en_dic = get_old_ens(this_params_list=this_params_list,
                                     conf_dic=conf_dic)

            print("Optimizing with tight thresholds...")
            tight_opt(params=params,
                      model=model,
                      conf_dic=conf_dic,
                      all_params=params_list,
                      num_parallel=num_parallel,
                      base_dir=base_dir,
                      params_list=params_list)

        except Exception as error:
            print("Failed with error '%s'" % error)
            is_done_list = update_w_error(this_params_list=this_params_list,
                                          conf_dic=conf_dic,
                                          error=error,
                                          is_done_list=is_done_list)
            continue

        is_done_list = update_is_done(conf_dic=conf_dic,
                                      old_en_dic=old_en_dic,
                                      params=params,
                                      is_done_list=is_done_list)

    os.chdir(cwd)
    params = params_list[0]
    print("Removing duplicates among all conformers...")
    batched_dedupe(conf_dic=conf_dic,
                   stage='tight',
                   params=params)

    print("Optimizing final CRE with ultra-tight thresholds...")

    vtight_opt(params=params,
               model=model,
               conf_dic=conf_dic,
               all_params=params_list,
               num_parallel=num_parallel,
               base_dir=base_dir,
               params_list=params_list)

    save_final_confs(conf_dic=conf_dic,
                     all_params=params_list)

    end = time.time()

    make_completion_files(base_dir=base_dir,
                          num_parallel=num_parallel,
                          start=start,
                          end=end)
    clean_and_summarize(all_params=params_list)

    print("Complete!")


if __name__ == "__main__":
    set_mtd_times(base_dir='.')
    main(base_dir='.')

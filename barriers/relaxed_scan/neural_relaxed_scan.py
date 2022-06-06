import numpy as np
import copy
import argparse
import json
from tqdm import tqdm
import time
import torch

from rdkit import Chem
from rdkit.Chem.rdMolTransforms import (SetDihedralDeg, SetAngleDeg, SetBondLength,
                                        GetDihedralDeg)


from ase.constraints import FixAtoms, Hookean
from ase.io.trajectory import TrajectoryWriter

from nff.io.ase_utils import ConstrainDihedrals, ConstrainAngles
from nff.io.ase import AtomsBatch
from barriers.utils.neuraloptimizer import (init_calculator, coords_to_xyz, opt_conformer,
                                            add_all_constraints, translate_template,
                                            get_model)
from barriers.confgen.neural_confgen import report_time

OPT_FILENAME = "opt.traj"


def rdkit_set_dihedral(atoms,
                       mol,
                       idx,
                       new_val):

    conf = mol.GetConformers()[0]
    SetDihedralDeg(conf,
                   idx[0],
                   idx[1],
                   idx[2],
                   idx[3],
                   new_val)
    pos = conf.GetPositions()
    atoms.set_positions(pos)

    return atoms, mol


def rdkit_set_angle(atoms,
                    mol,
                    idx,
                    new_val):

    conf = mol.GetConformers()[0]
    SetAngleDeg(conf,
                idx[0],
                idx[1],
                idx[2],
                new_val)
    pos = conf.GetPositions()
    atoms.set_positions(pos)

    return atoms, mol


def rdkit_set_bond_length(atoms,
                          mol,
                          idx,
                          new_val):

    conf = mol.GetConformers()[0]
    SetBondLength(conf,
                  idx[0],
                  idx[1],
                  new_val)
    pos = conf.GetPositions()
    atoms.set_positions(pos)

    return atoms, mol


def ase_set_dihedral(atoms,
                     mol,
                     idx,
                     new_val):

    # don't use `add=True` because then errors can accumulate
    # over steps in the relaxed scan

    # setting a negative value for the target dihedral automatically
    # gets 2 Pi added to it in ASE

    atoms.set_dihedral(idx[0],
                       idx[1],
                       idx[2],
                       idx[3],
                       new_val)
    return atoms, mol


def ase_set_angle(atoms,
                  idx,
                  mol,
                  new_val):

    atoms.set_angle(idx[0],
                    idx[1],
                    idx[2],
                    new_val)
    return atoms, mol


def ase_set_bond_length(atoms,
                        idx,
                        mol,
                        new_val):

    atoms.set_distance(idx[0],
                       idx[1],
                       new_val)
    return atoms, mol


def adjust_atoms(atoms,
                 params,
                 mol,
                 use_rdkit):
    """
    Adjust the positions of the atoms given the specified constraints on the
     internal degrees of freedom.

    `constraint_dic` is a dictionary with constraints, with specified targets
    for each internal, e.g.

    params = {"constraints": {"hookean": {"angles": {"idx": [[3, 4, 5], [5, 6, 7]],
                                           "targets": [[120.1, 140.3]],
                                           "force_const": 627.5},

                                "dihedrals": {"idx": null,
                                              "template_smiles": null,
                                              "targets": null,
                                              "force_const": 627.5}
                    }
        }
    }


    """

    rdkit_funcs = {"dihedrals": rdkit_set_dihedral,
                   "angles": rdkit_set_angle,
                   "bonds": rdkit_set_bond_length}

    ase_funcs = {"dihedrals": ase_set_dihedral,
                 "angles": ase_set_angle,
                 "bonds": ase_set_bond_length}

    dic = rdkit_funcs if (mol is not None and use_rdkit) else ase_funcs

    hookean = params["constraints"]['hookean']
    for internal_type, sub_dic in hookean.items():

        idx_set = sub_dic['idx']
        target_set = sub_dic['targets']
        func = dic[internal_type]

        if any([idx_set is None, target_set is None]):
            continue

        for idx, targets in zip(idx_set, target_set):
            atoms, mol = func(atoms=atoms,
                              idx=idx,
                              mol=mol,
                              new_val=targets)

    return atoms, mol


def get_mol(smiles):
    return Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))


def make_mol(atoms,
             smiles):

    if smiles is None:
        print(("No SMILES string supplied; using ASE to generate new geometry "
               "at each step instead of RDKit. Will likely require many more "
               "optimization steps because only the atoms constituting the internal "
               "degree of freedom are adjusted (e.g. the atoms in a dihedral angle). "
               "In RDKit all atoms are adjusted (e.g. those bonded to the atoms in the "
               "dihedral are rotated with it)."))
        return

    mol = get_mol(smiles)
    mol = Chem.AddHs(mol)
    conf = Chem.Conformer(len(atoms))
    pos = atoms.get_positions()

    for i, xyz in enumerate(pos):
        conf.SetAtomPosition(i, xyz)
    mol.AddConformer(conf)

    return mol


def adjust_end_dihed(initial,
                     end_val):

    # find convention that leads to the shortest path between initial and final
    possible_end_vals = np.array([end_val - 360, end_val, end_val + 360])
    closest_idx = np.argmin(abs(possible_end_vals - initial))
    new_end_val = possible_end_vals[closest_idx]

    return new_end_val


def get_new_vals(atoms,
                 internal_type,
                 end_val,
                 num_steps,
                 idx,
                 mol):

    if internal_type == 'bonds':
        initial = atoms.get_distance(idx[0],
                                     idx[1])
    elif internal_type == 'angles':
        initial = atoms.get_angle(idx[0],
                                  idx[1],
                                  idx[2])
    elif internal_type == 'dihedrals':

        # the value depends on the convention used by ASE or RDKit, which
        # may be different

        if mol is not None:
            initial = GetDihedralDeg(mol.GetConformers()[0], idx[0], idx[1],
                                     idx[2], idx[3])
        else:
            initial = atoms.get_dihedral(idx[0], idx[1], idx[2], idx[3])

        # adjust the end value to find the convention that gets the shortest path

        end_val = adjust_end_dihed(initial=initial,
                                   end_val=end_val)

    else:
        raise Exception(
            "Don't know how to handle internal type %s" % internal_type)

    # num_steps is the number of steps after the initial geometry
    delta = (end_val - initial) / num_steps
    new_vals = [initial + (i + 1) * delta for i in range(num_steps)]

    return new_vals


def get_constraint_list(atoms,
                        mol,
                        params,
                        num_steps):
    """
    Generate a set of constraints for each step along the relaxed scan. For example,
    given these params:


    params = {"end_constraints": {"hookean": {"atoms": {"idx": None,
                                              "template_smiles": None,
                                              "targets": None,
                                              "force_const": 2242.34},

                                   "bonds": {"idx": None,
                                             "template_smiles": None,
                                             "targets": None,
                                             "force_const": 2242.34},

                                   "angles": {"idx": [[4, 5, 6], [7, 8, 9]],
                                              "template_smiles": None,
                                              "targets": [130.2, 150.1],
                                              "force_const": 627.5},

                                   "dihedrals": {"idx": None,
                                                 "template_smiles": None,
                                                 "targets": None,
                                                 "force_const": 627.5}
                                   }
                       }
            },

    generate a copy for each step along the relaxed scan, but with the proper values
    of `targets` so that they interpolate from the initial values in the molecule
    to the final values specified in `end_constraints`.


    """

    # convert indices defined by any template SMILES to indices defined for
    # the current SMILES

    new_params = copy.deepcopy(params)
    # needs to be called `constraints` to get translated
    new_params['constraints'] = new_params['end_constraints']
    new_params = translate_template(new_params)

    # make the baseline constraint list, which we'll update with the appropriate
    # target values below

    hookean = new_params['constraints']['hookean']
    constraint_list = [copy.deepcopy(hookean) for _ in range(num_steps)]

    for internal_type, sub_dic in hookean.items():

        idx_set = sub_dic.get("idx")
        targets = sub_dic.get('targets')

        if idx_set is None:
            continue

        for i, idx in enumerate(idx_set):
            end_target = targets[i]
            scan_targets = get_new_vals(atoms=atoms,
                                        internal_type=internal_type,
                                        end_val=end_target,
                                        num_steps=num_steps,
                                        idx=idx,
                                        mol=mol)
            for j, scan_target in enumerate(scan_targets):
                constraint_list[j][internal_type]['targets'][i] = scan_target

    return constraint_list


def set_constraints(atoms,
                    fixed,
                    idx,
                    internal_type,
                    force_const):

    if fixed:
        c = FixAtoms(indices=idx)
        atoms.constraints = [c]
        return atoms

    assert force_const is not None, "Need to specify force constant if not fixing atoms"
    if internal_type == "angle":
        c = ConstrainAngles(idx=np.array([idx]),
                            atoms=atoms,
                            force_consts=np.array([force_const]))

    elif internal_type == "dihedral":
        c = ConstrainDihedrals(idx=np.array([idx]),
                               atoms=atoms,
                               force_consts=np.array([force_const]))

    elif internal_type == "bond_length":
        c = Hookean(a1=idx[0],
                    a2=idx[1],
                    k=force_const,
                    rt=0)

    else:
        raise Exception(("Internal type %s not supported; options are "
                         "'angle', 'dihedral', and 'bond_length'." % internal_type))

    atoms.constraints = [c]

    return atoms


def params_to_atoms(params, model):
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

    init_calculator(atoms=atoms,
                    params=params,
                    model=model)

    return atoms


def clone_atoms(atoms):
    """
    Copy atoms but delete the model in the calculator, so that we don't have to copy
    the model too
    """
    new_atoms = copy.deepcopy(atoms)
    new_atoms.calc.model = None

    return new_atoms


def idx_from_template(smiles,
                      ref_smiles,
                      ref_idx):

    mol = get_mol(smiles)
    ref_mol = get_mol(ref_smiles)

    idx_tuple = mol.GetSubstructMatch(ref_mol)
    substruc_idx = np.array(idx_tuple)
    reactive_idx = substruc_idx[np.array(ref_idx)].tolist()

    return reactive_idx


def tight_opt_conformer(atoms,
                        params,
                        model):
    new_params = copy.deepcopy(params)
    new_params["fmax"] = params["fmax_tight"]
    opt_conformer(atoms=atoms,
                  params=new_params,
                  model=model)


def save(trj):
    trj_writer = TrajectoryWriter(filename=OPT_FILENAME)
    for atoms in trj:
        trj_writer.write(atoms)
    trj_writer.close()


def run_opt(params):

    start = time.time()

    # Keep the atoms around so you only have to load the model for calc once,
    # as opposed to creating the atoms new every time

    model = get_model(params)
    atoms = params_to_atoms(params,
                            model=model)

    print("Optimizing input structure...")
    opt_conformer(atoms=atoms,
                  params=params,
                  model=model)
    print("Optimization complete!")

    num_steps = params["num_steps"]
    mol = make_mol(atoms=atoms,
                   smiles=params.get("smiles"))

    constraint_list = get_constraint_list(atoms=atoms,
                                          mol=mol,
                                          params=params,
                                          num_steps=num_steps)

    trj = [clone_atoms(atoms)]

    print("Performing a relaxed scan over %d steps..." % num_steps)

    # I've found that using RDKit can really mess things up, even though it should
    # in theory lead to faster convergence than ASE. So the default is not to use
    # RDKit
    use_rdkit = params.get("use_rdkit", False)

    for constraints in tqdm(constraint_list):

        new_params = copy.deepcopy(params)
        new_params['constraints'] = {'hookean': constraints}

        # adjust atoms to their ideal positions at this step in the relaxed scan
        adjust_atoms(atoms=atoms,
                     params=new_params,
                     mol=mol,
                     use_rdkit=use_rdkit)

        # add the Hookean constraints to enforce those values

        atoms.constraints = []
        add_all_constraints(atoms=atoms,
                            params=new_params)

        opt_conformer(atoms=atoms,
                      params=new_params,
                      model=model)

        trj.append(clone_atoms(atoms))

        save(trj)

    print("Performing final optimization with tight thresholds...")
    tight_opt_conformer(atoms=atoms,
                        params=new_params,
                        model=model)
    trj.append(atoms)
    print("Tight optimization complete!")

    save(trj)

    end = time.time()

    print("Relaxed scan complete! Trajectory saved to %s. " % OPT_FILENAME)

    report_time(start=start,
                end=end)


def update_params(params):
    """
    Get parameters that needed to be specified in `neuraloptimizer` that are now
    fixed in `neural_confgen`. Or params that have new names so they can be used
    with the old functions
    """

    new_params = {"do_md": True,
                  "do_save": True,
                  "assert_converged": False,
                  "max_rounds": 1,
                  "check_hess": False}

    params.update(new_params)


def combine_params(params):
    if 'details' in params:
        params.update(params['details'])
        params.pop('details')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--info_file',
                        help=("Name of file with details for the job"),
                        default='job_info.json')
    args = parser.parse_args()

    with open(args.info_file, 'r') as f:
        params = json.load(f)

    combine_params(params)
    update_params(params)
    run_opt(params)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()

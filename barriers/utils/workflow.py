"""
Script for patching together all the stages of TS generation and singlet/triplet
crossings.
"""

import os
import json
import pickle
import numpy as np
import copy
from tqdm import tqdm
from rdkit import Chem
import argparse

from ase.io.trajectory import Trajectory

from nff.utils.confgen import get_mol
from nff.utils.misc import bash_command
from barriers.confgen.neural_confgen import atoms_to_nxyz
from barriers.utils.neuraloptimizer import get_substruc_idx

ANGLE_CONSTRAINTS = {'constrain_rot': {
    "idx": [[3, 4, 5], [4, 5, 6]],
    "template_smiles": "c1ccc(/N=N/c2ccccc2)cc1",
    "targets": [122.0, 122.0],
    "force_consts": 627.5
},
    'left_invert': {
    "idx": [[3, 4, 5]],
    "template_smiles": "c1ccc(/N=N/c2ccccc2)cc1",
    "targets": [179.5],
    "force_consts": 627.5
},
    'right_invert': {
    "idx": [[4, 5, 6]],
    "template_smiles": "c1ccc(/N=N/c2ccccc2)cc1",
    "targets": [179.5],
    "force_consts": 627.5
}}

DIHED_CONSTRAINTS = {'left_rot': {
    "idx": [[3, 4, 5, 6]],
    "template_smiles": "c1ccc(/N=N/c2ccccc2)cc1",
    "targets": [90.0],
    "force_consts": 627.5
},
    'right_rot': {
    "idx": [[3, 4, 5, 6]],
    "template_smiles": "c1ccc(/N=N/c2ccccc2)cc1",
    "targets": [-90.0],
    "force_consts": 627.5
}}

MECH_CONSTRAINTS = [{"angles": ANGLE_CONSTRAINTS['constrain_rot'],
                     "dihedrals": DIHED_CONSTRAINTS['left_rot']},
                    {"angles": ANGLE_CONSTRAINTS['constrain_rot'],
                     "dihedrals": DIHED_CONSTRAINTS['right_rot']},
                    {"angles": ANGLE_CONSTRAINTS['left_invert']},
                    {"angles": ANGLE_CONSTRAINTS['right_invert']}]

NAME_LIST = ['left_rot', 'right_rot', 'left_invert', 'right_invert']

AZO_FIXED_ATOMS = {"idx": [3, 4, 5, 6],
                   "template_smiles": "c1ccc(/N=N/c2ccccc2)cc1"}

TRANS_AZO = "c1ccc(/N=N/c2ccccc2)cc1"
CIS_AZO = "c1ccc(/N=N\\c2ccccc2)cc1"
SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '../../scripts')


def rd_to_nxyz(rd_mol):

    n = np.array([i.GetAtomicNum() for i in rd_mol.GetAtoms()])
    xyz = np.array(rd_mol.GetConformers()[0].GetPositions())
    nxyz = np.concatenate([n.reshape(-1, 1), xyz], axis=-1).tolist()

    return nxyz


def load_rdkit_confgen(rd_dir,
                       only_trans):

    info_list = []
    files = [i for i in os.listdir(rd_dir) if i.endswith("pickle")]

    for file in tqdm(files):
        path = os.path.join(rd_dir, file)
        with open(path, 'rb') as f:
            dic = pickle.load(f)

        if 'conformers' not in dic or 'smiles' not in dic:
            continue

        rd_mol = dic['conformers'][0]['rd_mol']
        smiles = dic['smiles']

        mol_no_h = get_mol(smiles)

        if only_trans:
            template_mol = Chem.MolFromSmiles(TRANS_AZO)
            keep = mol_no_h.HasSubstructMatch(template_mol,
                                              useChirality=True)
            if not keep:
                continue

        info = {"nxyz": rd_to_nxyz(rd_mol),
                "smiles": smiles,
                "inchikey": Chem.inchi.MolToInchiKey(mol_no_h)}

        info_list.append(info)

    return info_list


def make_scan_info_list(rd_dir,
                        base_info):
    info_list = load_rdkit_confgen(rd_dir=rd_dir,
                                   only_trans=True)
    new_info_list = []
    for info in info_list:
        for constraints in MECH_CONSTRAINTS:
            new_info = copy.deepcopy(base_info)
            new_info.update(info)
            new_info.update({"end_constraints": {"hookean": constraints}})

            if isinstance(base_info.get("relaxed_scan"), dict):
                new_info.update(base_info['relaxed_scan'])

            new_info_list.append(new_info)

    return new_info_list


def make_scan_sub_dir(info,
                      inchi_count,
                      scan_dir):

    inchikey = info['inchikey']
    if inchikey not in inchi_count:
        inchi_count[inchikey] = -1

    inchi_count[inchikey] += 1
    mech_name = NAME_LIST[inchi_count[inchikey]]
    folder_name = "%s_%s" % (inchikey, mech_name)
    direc = os.path.join(scan_dir, folder_name)

    if not os.path.isdir(direc):
        os.makedirs(direc)

    save_path = os.path.join(direc, 'job_info.json')
    info.update({"mechanism": mech_name})
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=4)

    base_job_path = os.path.join(SCRIPTS, "relaxed_scan/job.sh")
    new_job_file = os.path.join(direc, 'job.sh')

    text = "bash %s\n" % base_job_path
    with open(new_job_file, 'w') as f:
        f.write(text)


def render_general_batch(direc, config_name):
    base_batch_path = os.path.join(SCRIPTS, "%s/batch.sh" % config_name)
    new_batch_path = os.path.join(direc, 'batch.sh')
    batch_text = 'bash %s\n' % base_batch_path

    with open(new_batch_path, 'w') as f:
        f.write(batch_text)


def rdkit_to_scan_dirs(rd_dir,
                       scan_dir,
                       base_info):
    """
    Take the results of RDKit conformer generation and use them as input to a relaxed
    scan job.
    """

    new_info_list = make_scan_info_list(rd_dir=rd_dir,
                                        base_info=base_info)

    inchi_count = {}
    for info in new_info_list:
        make_scan_sub_dir(info=info,
                          inchi_count=inchi_count,
                          scan_dir=scan_dir)

    render_general_batch(direc=scan_dir,
                         config_name='relaxed_scan')


def make_ts_confgen_info_list(base_info,
                              scan_dir):

    info_list = []
    folders = [os.path.join(scan_dir, i) for i in os.listdir(scan_dir)]

    for folder in folders:
        if not os.path.isdir(folder):
            continue
        opt_path = os.path.join(folder, 'opt.traj')
        if not os.path.isfile(opt_path):
            continue

        info_path = os.path.join(folder, 'job_info.json')
        if not os.path.isfile(info_path):
            continue

        with open(info_path, 'r') as f:
            old_info = json.load(f)

        atoms = Trajectory(opt_path)[-1]
        nxyz = atoms_to_nxyz(atoms)

        inherit_keys = ['smiles', 'inchikey', 'mechanism']

        new_info = copy.deepcopy(base_info)
        new_info.update({key: old_info[key] for key in inherit_keys})
        new_info.update({'nxyz': nxyz,
                         'fixed_atoms': AZO_FIXED_ATOMS})

        if isinstance(base_info.get("confgen"), dict):
            new_info.update(base_info['confgen'])

        info_list.append(new_info)

    return info_list


def make_confgen_sub_dir(info,
                         confgen_dir):

    inchikey = info['inchikey']
    mech_name = info['mechanism']
    folder_name = "%s_%s" % (inchikey, mech_name)
    direc = os.path.join(confgen_dir, folder_name)

    if not os.path.isdir(direc):
        os.makedirs(direc)

    save_path = os.path.join(direc, 'job_info.json')
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=4)

    base_job_path = os.path.join(SCRIPTS, "confgen/job.sh")
    new_job_file = os.path.join(direc, 'job.sh')

    text = "bash %s\n" % base_job_path
    with open(new_job_file, 'w') as f:
        f.write(text)


def scan_to_confgen_dirs(scan_dir,
                         confgen_dir,
                         base_info):
    """
    Take the results of RDKit conformer generation and use them as input to a relaxed
    scan job.
    """

    new_info_list = make_ts_confgen_info_list(base_info=base_info,
                                              scan_dir=scan_dir)

    for info in new_info_list:
        make_confgen_sub_dir(info=info,
                             confgen_dir=confgen_dir)

    render_general_batch(direc=confgen_dir,
                         config_name='confgen')


def rdkit_to_neural_confgen_dirs():
    """
    Take the results of RDKit conformer generation and use them as input to a neural
    confgen job to optimize the cis and trans endpoints.
    """
    pass


def bond_idx_match(bond,
                   idx):

    start_idx = bond.GetBeginAtomIdx()
    end_idx = bond.GetEndAtomIdx()

    match = set([start_idx, end_idx]) == set(idx)

    return match


def make_cis_trans(smiles_list):

    full_smiles_list = []
    for smiles in smiles_list:
        template_mol = Chem.MolFromSmiles(TRANS_AZO)
        mol_no_h = Chem.MolFromSmiles(smiles)
        keep = mol_no_h.HasSubstructMatch(template_mol,
                                          useChirality=False)
        if not keep:
            print(("Smiles %s doesn't have an azobenzene substructure; skipping" %
                   smiles))
            continue

        substruc_idx = get_substruc_idx(template_smiles=TRANS_AZO,
                                        smiles=smiles)
        nn_idx = substruc_idx[4:6]

        isomers = [Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE]
        for bond_iso in isomers:
            new_mol = Chem.MolFromSmiles(smiles)
            nn_bond_pairs = [[i, b] for i, b in enumerate(new_mol.GetBonds()) if
                             bond_idx_match(bond=b, idx=nn_idx)]
            if len(nn_bond_pairs) != 1:
                print(("Problem finding the N=N bond in the azobenzene subtstructure "
                       "of smiles %s. Skipping" % smiles))
                continue

            bond_idx, bond = nn_bond_pairs[0]
            bond.SetStereo(bond_iso)

            new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(
                Chem.MolToSmiles(
                    new_mol
                )))
            full_smiles_list.append(new_smiles)

    return full_smiles_list


def make_rdkit_dir(rd_dir,
                   base_info):
    smiles_list = base_info['smiles_list']
    smiles_list = make_cis_trans(smiles_list)

    csv_path = os.path.join(rd_dir, 'smiles.csv')
    text = "smiles\n"
    for smiles in smiles_list:
        text += "%s\n" % smiles

    with open(csv_path, 'w') as f:
        f.write(text)

    info_path = os.path.join(rd_dir, 'job_info.json')
    info = copy.deepcopy(base_info)
    if isinstance(base_info.get("rdkit_confgen"), dict):
        info.update(base_info['rdkit_confgen'])

    # remove smiles_list from info because it leads to weird conflicts
    # with loading the smiles from csv

    if 'smiles_list' in info:
        info.pop('smiles_list')

    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)


def make_all_subdirs(base_dir):
    names = ['rdkit_confgen', 'relaxed_scan', 'confgen', 'evf', 'irc']
    dic_info = {name: os.path.join(base_dir, 'results', name)
                for name in names}

    for direc in dic_info.values():
        if os.path.isdir(direc):
            continue
        os.makedirs(direc)

    return dic_info


def run_stage(stage,
              dir_info,
              base_dir,
              do_batch):

    print("Running %s stage..." % stage)

    sub_dir = dir_info[stage]
    job_file = 'batch.sh' if do_batch else 'job.sh'
    job_script = os.path.join(SCRIPTS, stage, job_file)
    cmd = "cd %s && bash %s" % (sub_dir, job_script)
    p = bash_command(cmd)
    p.wait()

    os.chdir(base_dir)

    print("%s complete!" % stage)


def run_all(base_dir):
    info_path = os.path.join(base_dir, 'job_info.json')
    with open(info_path, 'r') as f:
        base_info = json.load(f)

    dir_info = make_all_subdirs(base_dir=base_dir)
    make_rdkit_dir(rd_dir=dir_info['rdkit_confgen'],
                   base_info=base_info)

    run_stage(stage='rdkit_confgen',
              dir_info=dir_info,
              base_dir=base_dir,
              do_batch=False,)

    rdkit_to_scan_dirs(rd_dir=dir_info['rdkit_confgen'],
                       scan_dir=dir_info['relaxed_scan'],
                       base_info=base_info)

    run_stage(stage='relaxed_scan',
              dir_info=dir_info,
              base_dir=base_dir,
              do_batch=True)

    scan_to_confgen_dirs(scan_dir=dir_info['relaxed_scan'],
                         confgen_dir=dir_info['confgen'],
                         base_info=base_info)

    run_stage(stage='confgen',
              dir_info=dir_info,
              base_dir=base_dir,
              do_batch=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",
                        type=str,
                        help="Where to run the calculations",
                        default=".")
    args = parser.parse_args()

    run_all(base_dir=args.base_dir)


if __name__ == "__main__":
    main()

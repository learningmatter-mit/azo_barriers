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
from rdkit.Chem import inchi
from ase.io.trajectory import Trajectory

from nff.utils.confgen import get_mol
from barriers.confgen.neural_confgen import atoms_to_nxyz

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

SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '../../scripts')


def rd_to_nxyz(rd_mol):

    n = np.array([i.GetAtomicNum() for i in rd_mol.GetAtoms()])
    xyz = np.array(rd_mol.GetConformers()[0].GetPositions())
    nxyz = np.concatenate([n.reshape(-1, 1), xyz], axis=-1).tolist()

    return nxyz


def load_rdkit_confgen(rd_dir):

    info_list = []
    files = [i for i in os.listdir(rd_dir) if i.endswith("pickle")]

    for file in tqdm(files):
        path = os.path.join(rd_dir, file)
        with open(path, 'rb') as f:
            dic = pickle.load(f)

        if 'conformers' not in dic or 'smiles' not in dic:
            continue

        rd_mol = dic['conformers'][0]['rd_mol']
        info = {"nxyz": rd_to_nxyz(rd_mol),
                "smiles": dic['smiles'],
                "inchikey": inchi.MolToInchiKey(get_mol(dic['smiles']))}

        info_list.append(info)

    return info_list


def make_scan_info_list(rd_dir,
                        base_info):
    info_list = load_rdkit_confgen(rd_dir)
    new_info_list = []
    for info in info_list:
        for constraints in MECH_CONSTRAINTS:
            new_info = copy.deepcopy(base_info)
            new_info.update(info)
            new_info.update({"end_constraints": {"hookean": constraints}})
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

"""
Script for patching together all the stages of TS generation and singlet/triplet
crossings.
"""

import os
import json
import pickle
import numpy as np
import copy
import shutil
from tqdm import tqdm
from rdkit.Chem import inchi
from nff.utils.confgen import get_mol

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


def rdkit_to_scan_dirs(rd_dir,
                       scan_dir,
                       base_info):
    """
    Take the results of RDKit conformer generation and use them as input to a relaxed
    scan job.
    """

    info_list = load_rdkit_confgen(rd_dir)
    new_info_list = []
    for info in info_list:
        for constraints in MECH_CONSTRAINTS:
            new_info = copy.deepcopy(base_info)
            new_info.update(info)
            new_info.update({"end_constraints": {"hookean": constraints}})
            new_info_list.append(new_info)

    inchi_count = {}
    for info in new_info_list:
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
        with open(save_path, 'w') as f:
            json.dump(info, f, indent=4)

        base_job_path = os.path.join(SCRIPTS, "relaxed_scan/job.sh")
        new_job_file = os.path.join(direc, 'job.sh')

        text = "bash %s\n" % base_job_path
        with open(new_job_file, 'w') as f:
            f.write(text)

    base_batch_path = os.path.join(SCRIPTS, "relaxed_scan/batch.sh")
    new_batch_path = os.path.join(scan_dir, 'batch.sh')
    batch_text = 'bash %s\n' % base_batch_path

    with open(new_batch_path, 'w') as f:
        f.write(batch_text)


def rdkit_to_neural_confgen_dirs():
    """
    Take the results of RDKit conformer generation and use them as input to a neural
    confgen job to optimize the cis and trans endpoints.
    """
    pass

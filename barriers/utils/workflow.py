"""
Script for patching together all the stages of TS generation and singlet/triplet
crossings.
"""

import os
import json
import pickle
import numpy as np
import copy
from rdkit import Chem
import argparse

from ase.io.trajectory import Trajectory

from nff.utils.confgen import get_mol
from nff.utils.misc import bash_command
from nff.utils import constants as const

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
CONFIG_NAMES = ['rdkit_confgen',
                'relaxed_scan',
                'confgen',
                'evf',
                'irc',
                'triplet_crossing',
                'hessian']

ENTROPY_CONV = 8805.96228743921
TEMP = 298.15
KB_HA = 3.167e-6
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

    for file in files:
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


def make_confgen_info_list(base_info,
                           scan_dir,
                           rd_dir):

    folders = [os.path.join(scan_dir, i) for i in os.listdir(scan_dir)]

    info_list = []

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
                         'fixed_atoms': AZO_FIXED_ATOMS,
                         'geom_type': "ts"})

        if isinstance(base_info.get("confgen"), dict):
            new_info.update(base_info['confgen'])

        info_list.append(new_info)

    # for endpoints' confgen
    rd_info_list = load_rdkit_confgen(rd_dir=rd_dir,
                                      only_trans=False)
    for old_info in rd_info_list:
        new_info = copy.deepcopy(base_info)
        # includes xyz
        new_info.update(old_info)
        new_info.update({'exclude_from_rmsd': AZO_FIXED_ATOMS,
                         'fixed_atoms': {"idx": None,
                                         "template_smiles": None},
                         "geom_type": "endpoint"})

        if isinstance(base_info.get("confgen"), dict):
            new_info.update(base_info['confgen'])

        info_list.append(new_info)

    return info_list


def make_confgen_sub_dir(info,
                         confgen_dir):

    inchikey = info['inchikey']

    if 'mechanism' in info and info['geom_type'] == 'ts':
        mech_name = info['mechanism']
        folder_name = "%s_%s" % (inchikey, mech_name)
    else:
        folder_name = "%s_endpoint" % inchikey

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
                         base_info,
                         rd_dir):

    new_info_list = make_confgen_info_list(base_info=base_info,
                                           scan_dir=scan_dir,
                                           rd_dir=rd_dir)

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


def make_evf_info_list(base_info,
                       confgen_dir):

    folders = [os.path.join(confgen_dir, i) for i in os.listdir(confgen_dir)]
    confs_per_ts = base_info.get("evf", {}).get("confs_per_ts")
    if confs_per_ts is None:
        confs_per_ts = 5

    info_list = []

    for folder in folders:
        if not os.path.isdir(folder):
            continue
        opt_path = os.path.join(folder, 'final_opt.traj')
        if not os.path.isfile(opt_path):
            continue

        info_path = os.path.join(folder, 'job_info.json')
        if not os.path.isfile(info_path):
            continue

        with open(info_path, 'r') as f:
            old_info = json.load(f)

        geom_type = old_info['geom_type']
        if geom_type != "ts":
            continue

        trj = Trajectory(opt_path)
        for i, atoms in enumerate(trj[:confs_per_ts]):
            nxyz = atoms_to_nxyz(atoms)
            confnum = i + 1

            inherit_keys = ['smiles', 'inchikey', 'mechanism']

            new_info = copy.deepcopy(base_info)
            new_info.update({key: old_info[key] for key in inherit_keys})
            new_info.update({'nxyz': nxyz,
                             'confnum': confnum})

            if isinstance(base_info.get("evf"), dict):
                new_info.update(base_info['evf'])

            info_list.append(new_info)

    return info_list


def make_evf_sub_dir(info,
                     evf_dir):

    inchikey = info['inchikey']
    mech_name = info['mechanism']
    confnum = info['confnum']

    folder_name = "%s_%s_conf_%d" % (inchikey, mech_name, confnum)
    direc = os.path.join(evf_dir, folder_name)

    if not os.path.isdir(direc):
        os.makedirs(direc)

    save_path = os.path.join(direc, 'job_info.json')
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=4)

    base_job_path = os.path.join(SCRIPTS, "evf/job.sh")
    new_job_file = os.path.join(direc, 'job.sh')

    text = "bash %s\n" % base_job_path
    with open(new_job_file, 'w') as f:
        f.write(text)


def confgen_to_evf_dirs(confgen_dir,
                        evf_dir,
                        base_info):

    new_info_list = make_evf_info_list(base_info=base_info,
                                       confgen_dir=confgen_dir)

    for info in new_info_list:
        make_evf_sub_dir(info=info,
                         evf_dir=evf_dir)

    render_general_batch(direc=evf_dir,
                         config_name='evf')


def make_hess_info_list(base_info,
                        confgen_dir):

    folders = [os.path.join(confgen_dir, i) for i in os.listdir(confgen_dir)]
    confs_per_endpoint = base_info.get("hessian", {}).get("confs_per_endpoint")
    if confs_per_endpoint is None:
        confs_per_endpoint = 1

    info_list = []

    for folder in folders:
        if not os.path.isdir(folder):
            continue

        info_path = os.path.join(folder, 'job_info.json')
        if not os.path.isfile(info_path):
            continue

        with open(info_path, 'r') as f:
            old_info = json.load(f)

        geom_type = old_info['geom_type']
        if geom_type != "endpoint":
            continue

        opt_path = os.path.join(folder, 'final_opt.traj')
        if not os.path.isfile(opt_path):
            continue

        trj = Trajectory(opt_path)
        for i, atoms in enumerate(trj[:confs_per_endpoint]):
            nxyz = atoms_to_nxyz(atoms)
            confnum = i + 1

            inherit_keys = ['smiles', 'inchikey']

            new_info = copy.deepcopy(base_info)
            new_info.update({key: old_info[key] for key in inherit_keys})
            new_info.update({'nxyz': nxyz,
                             'confnum': confnum})

            if isinstance(base_info.get("hessian"), dict):
                new_info.update(base_info['hessian'])

            info_list.append(new_info)

    return info_list


def make_hessian_sub_dir(info,
                         hess_dir):

    inchikey = info['inchikey']
    confnum = info['confnum']

    folder_name = "%s_endpoint_conf_%d" % (inchikey, confnum)
    direc = os.path.join(hess_dir, folder_name)

    if not os.path.isdir(direc):
        os.makedirs(direc)

    save_path = os.path.join(direc, 'job_info.json')
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=4)

    base_job_path = os.path.join(SCRIPTS, "hessian/job.sh")
    new_job_file = os.path.join(direc, 'job.sh')

    text = "bash %s\n" % base_job_path
    with open(new_job_file, 'w') as f:
        f.write(text)


def confgen_to_hessian_dirs(confgen_dir,
                            hess_dir,
                            base_info):

    new_info_list = make_hess_info_list(base_info=base_info,
                                        confgen_dir=confgen_dir)

    for info in new_info_list:
        make_hessian_sub_dir(info=info,
                             hess_dir=hess_dir)

    render_general_batch(direc=hess_dir,
                         config_name='neuralhessian')


def make_irc_info_list(base_info,
                       evf_dir,
                       mech_key=None):

    folders = [os.path.join(evf_dir, i) for i in os.listdir(evf_dir)]
    confs_per_ts = base_info.get("irc", {}).get("confs_per_ts")
    if confs_per_ts is None:
        confs_per_ts = 1

    # load in all results from each conformer for each species

    info_dic = {}

    for folder in folders:
        if not os.path.isdir(folder):
            continue
        ts_path = os.path.join(folder, 'ts.pickle')
        if not os.path.isfile(ts_path):
            continue

        info_path = os.path.join(folder, 'job_info.json')
        if not os.path.isfile(info_path):
            continue

        with open(info_path, 'r') as f:
            old_info = json.load(f)

        with open(ts_path, 'rb') as f:
            dic = pickle.load(f)

        if not dic['converged']:
            print("Skipping %s since the TS is not converged" % folder)
            continue

        mech = old_info['mechanism']
        if mech_key is not None:
            if mech_key not in mech:
                continue

        inherit_keys = ['smiles', 'inchikey', 'mechanism', 'confnum']

        new_info = copy.deepcopy(base_info)
        new_info.update({key: old_info[key] for key in inherit_keys})
        new_info.update({'nxyz': dic['nxyz'],
                         'freeenergy': dic['freeenergy']})

        inchikey = old_info['inchikey']

        if inchikey not in info_dic:
            info_dic[inchikey] = {}

        if mech not in info_dic[inchikey]:
            info_dic[inchikey][mech] = []

        info_dic[inchikey][mech].append(new_info)

    # chose the `confs_per_ts` evf geometries of lowest free energy for each
    # mechanism

    info_list = []
    for mech_dic in info_dic.values():
        results_by_mech = list(mech_dic.values())
        for mech_results in results_by_mech:
            if not mech_results:
                continue
            sort_results = list(sorted(mech_results,
                                       key=lambda x: x['freeenergy']))
            info_list += sort_results[:confs_per_ts]

    return info_list


def make_irc_sub_dir(info,
                     irc_dir,
                     dir_name='irc'):

    inchikey = info['inchikey']
    mech_name = info['mechanism']
    confnum = info['confnum']

    folder_name = "%s_%s_conf_%d" % (inchikey, mech_name, confnum)
    direc = os.path.join(irc_dir, folder_name)

    if not os.path.isdir(direc):
        os.makedirs(direc)

    save_path = os.path.join(direc, 'job_info.json')
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=4)

    base_job_path = os.path.join(SCRIPTS, "%s/job.sh" % dir_name)
    new_job_file = os.path.join(direc, 'job.sh')

    text = "bash %s\n" % base_job_path
    with open(new_job_file, 'w') as f:
        f.write(text)


def evf_to_irc_dirs(evf_dir,
                    irc_dir,
                    base_info):

    new_info_list = make_irc_info_list(base_info=base_info,
                                       evf_dir=evf_dir)

    for info in new_info_list:
        make_irc_sub_dir(info=info,
                         irc_dir=irc_dir)

    render_general_batch(direc=irc_dir,
                         config_name='irc')


def make_triplet_info_list(base_info,
                           evf_dir):

    results = make_irc_info_list(base_info=base_info,
                                 evf_dir=evf_dir,
                                 mech_key='rot')

    return results


def make_triplet_sub_dir(info,
                         triplet_dir):

    out = make_irc_sub_dir(info=info,
                           irc_dir=triplet_dir,
                           dir_name='triplet_crossing')

    return out


def evf_to_triplet_dirs(evf_dir,
                        triplet_dir,
                        base_info):

    new_info_list = make_triplet_info_list(base_info=base_info,
                                           evf_dir=evf_dir)

    for info in new_info_list:
        make_triplet_sub_dir(info=info,
                             triplet_dir=triplet_dir)

    render_general_batch(direc=triplet_dir,
                         config_name='triplet_crossing')


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
    dic_info = {name: os.path.join(base_dir, 'results', name)
                for name in CONFIG_NAMES}

    for direc in dic_info.values():
        if os.path.isdir(direc):
            continue
        os.makedirs(direc)

    return dic_info


def run_stage(stage,
              dir_info,
              base_dir,
              do_batch,
              description=None):

    if description is None:
        description = stage

    print("Running %s..." % description)

    sub_dir = dir_info[stage]
    job_file = 'batch.sh' if do_batch else 'job.sh'
    job_script = os.path.join(SCRIPTS, stage, job_file)
    cmd = "cd %s && bash %s" % (sub_dir, job_script)
    p = bash_command(cmd)
    p.wait()

    os.chdir(base_dir)

    print("%s complete!" % stage)


def run_rdkit(dir_info,
              base_info,
              base_dir):
    make_rdkit_dir(rd_dir=dir_info['rdkit_confgen'],
                   base_info=base_info)

    run_stage(stage='rdkit_confgen',
              dir_info=dir_info,
              base_dir=base_dir,
              do_batch=False,
              description=('RDKit conformer generation jobs on '
                           'cis and trans azobenzene derivatives'))


def run_relaxed_scan(dir_info,
                     base_info,
                     base_dir):

    rdkit_to_scan_dirs(rd_dir=dir_info['rdkit_confgen'],
                       scan_dir=dir_info['relaxed_scan'],
                       base_info=base_info)

    run_stage(stage='relaxed_scan',
              dir_info=dir_info,
              base_dir=base_dir,
              do_batch=True,
              description=('relaxed scans from RDKit trans conformers to '
                           'TSs for each of the four mechanisms'))


def run_confgen(dir_info,
                base_info,
                base_dir):

    scan_to_confgen_dirs(scan_dir=dir_info['relaxed_scan'],
                         confgen_dir=dir_info['confgen'],
                         base_info=base_info,
                         rd_dir=dir_info['rdkit_confgen'])

    run_stage(stage='confgen',
              dir_info=dir_info,
              base_dir=base_dir,
              do_batch=True,
              description=('conformer generation on the TSs, the reactants, '
                           'and the products.'))


def run_evf(dir_info,
            base_info,
            base_dir):

    confgen_to_evf_dirs(confgen_dir=dir_info['confgen'],
                        evf_dir=dir_info['evf'],
                        base_info=base_info)

    run_stage(stage='evf',
              dir_info=dir_info,
              base_dir=base_dir,
              do_batch=True,
              description=('eigenvector following on TS conformers'))


def run_hess(dir_info,
             base_info,
             base_dir):

    confgen_to_hessian_dirs(confgen_dir=dir_info['confgen'],
                            hess_dir=dir_info['hessian'],
                            base_info=base_info)

    run_stage(stage='hessian',
              dir_info=dir_info,
              base_dir=base_dir,
              do_batch=True,
              description='hessian calculations on optimized reactants and products')


def run_triplet(dir_info,
                base_info,
                base_dir):

    evf_to_triplet_dirs(evf_dir=dir_info['evf'],
                        triplet_dir=dir_info['triplet_crossing'],
                        base_info=base_info)

    run_stage(stage='triplet_crossing',
              dir_info=dir_info,
              base_dir=base_dir,
              do_batch=True,
              description='singlet-triplet crossing optimizations')


def run_irc(dir_info,
            base_info,
            base_dir):

    evf_to_irc_dirs(evf_dir=dir_info['evf'],
                    irc_dir=dir_info['irc'],
                    base_info=base_info)

    run_stage(stage='irc',
              dir_info=dir_info,
              base_dir=base_dir,
              do_batch=True,
              description=('intrinsic reaction coordinate calculations on optimized '
                           'transition states'))


def run_simulations(**kwargs):
    funcs = [run_rdkit, run_relaxed_scan, run_confgen, run_hess,
             run_evf, run_triplet, run_irc]

    for func in funcs:
        func(**kwargs)


def get_conf_g(confgen_sub_dir):

    path = os.path.join(confgen_sub_dir, 'final_opt.traj')
    trj = Trajectory(path)
    ens = np.array([float(i.get_potential_energy()) for i in trj])

    kt = KB_HA * TEMP
    rel_ens = (ens - min(ens)) * const.EV_TO_AU
    p = np.exp(-rel_ens / kt)
    p /= p.sum()

    conf_s = KB_HA * (-p * np.log(p)).sum()
    delta_mean_e = (rel_ens * p).sum()
    conf_g = -TEMP * conf_s + delta_mean_e

    return conf_g


def update_w_conf_g(conf_g,
                    dic):

    dic.update({"free_energy": dic['free_energy'] + conf_g,
                "conf_free_energy": conf_g,
                # "free_energy_no_conf": dic['free_energy'],
                "entropy": dic['entropy'] - conf_g,
                "vib_entropy": dic['entropy'],
                "conf_entropy": -conf_g})


def make_ts_summary(ts_sub_dir):

    ts_path = os.path.join(ts_sub_dir, 'ts.pickle')
    if not os.path.isfile(ts_path):
        return

    info_path = os.path.join(ts_sub_dir, 'job_info.json')
    if not os.path.isfile(info_path):
        return

    with open(info_path, 'r') as f:
        old_info = json.load(f)

    with open(ts_path, 'rb') as f:
        dic = pickle.load(f)

    ts_keys = ['converged', 'entropy', 'freeenergy', 'enthalpy', 'vibfreqs',
               'energy', 'nxyz']
    translate = {"freeenergy": "free_energy"}
    ts_summary = {"smiles": old_info['smiles'],
                  "mechanism": old_info["mechanism"]}

    for key in ts_keys:
        if key not in dic:
            continue
        ts_summary[translate.get(key, key)] = dic[key]

    # add conformational free energy
    confgen_sub_dir = ts_sub_dir.replace("evf", "confgen").split("_conf_")[0]
    conf_g = get_conf_g(confgen_sub_dir=confgen_sub_dir)
    update_w_conf_g(conf_g=conf_g,
                    dic=ts_summary)

    return ts_summary


def make_cis(smiles):
    substruc_idx = get_substruc_idx(template_smiles=TRANS_AZO,
                                    smiles=smiles)
    nn_idx = substruc_idx[4:6]
    new_mol = Chem.MolFromSmiles(smiles)
    nn_bond_pairs = [[i, b] for i, b in enumerate(new_mol.GetBonds()) if
                     bond_idx_match(bond=b, idx=nn_idx)]
    if len(nn_bond_pairs) != 1:
        print(("Problem finding the N=N bond in the azobenzene subtstructure "
               "of smiles %s. Skipping" % smiles))
        return

    bond_idx, bond = nn_bond_pairs[0]
    bond.SetStereo(Chem.BondStereo.STEREOZ)

    cis_smiles = Chem.MolToSmiles(
        Chem.MolFromSmiles(
            Chem.MolToSmiles(
                new_mol
            )
        )
    )

    return cis_smiles


def summarize_all_ts(ts_dir,
                     final_info_dict):

    for i in os.listdir(ts_dir):
        ts_sub_dir = os.path.join(ts_dir, i)
        if not os.path.isdir(ts_sub_dir):
            continue
        ts_summary = make_ts_summary(ts_sub_dir=ts_sub_dir)
        if ts_summary is None:
            continue

        smiles = make_cis(ts_summary['smiles'])
        if smiles is None:
            continue

        ts_summary.pop('smiles')
        if smiles not in final_info_dict:
            final_info_dict[smiles] = {}

        sub_dic = final_info_dict[smiles]
        if 'transition_states' not in sub_dic:
            sub_dic['transition_states'] = {}

        ts_dic = sub_dic['transition_states']
        mech = ts_summary['mechanism']
        if mech not in ts_dic:
            ts_dic[mech] = []

        ts_dic[mech].append(ts_summary)

    # convert `transition_states` to a list
    for sub_dic in final_info_dict.values():
        sub_dic['transition_states'] = list(sub_dic['transition_states']
                                            .values())


def filter_by_done(final_info_dict):
    keep_keys = []
    for key, sub_dic in final_info_dict.items():
        # check that all the mechanisms finished and keep only the converged ones
        ts_lists = sub_dic['transition_states']
        keep_ts_lists = []
        num_done = 0
        for ts_list in ts_lists:
            if len(ts_list) > 0:
                num_done += 1
            new_ts_list = [dic for dic in ts_list if dic['converged']]
            if new_ts_list:
                keep_ts_lists.append(new_ts_list)

        sub_dic['transition_states'] = keep_ts_lists
        if num_done >= 4:
            keep_keys.append(key)

    final_info_dict = {key: final_info_dict[key] for key in keep_keys}

    return final_info_dict


def make_end_summary(hess_sub_dir):

    hess_path = os.path.join(hess_sub_dir, 'hess.json')
    if not os.path.isfile(hess_path):
        return

    info_path = os.path.join(hess_sub_dir, 'job_info.json')
    if not os.path.isfile(info_path):
        return

    with open(info_path, 'r') as f:
        old_info = json.load(f)

    with open(hess_path, 'rb') as f:
        dic = json.load(f)

    ts_keys = ['converged', 'entropy', 'freeenergy', 'enthalpy', 'vibfreqs',
               'energy']
    translate = {"freeenergy": "free_energy"}
    imgfreq = len([i for i in dic['vibfreqs'] if i < old_info['imag_cutoff']])
    converged = (imgfreq == 0)

    hess_summary = {"smiles": old_info['smiles'],
                    'nxyz': old_info['nxyz'],
                    "converged": converged}

    for key in ts_keys:
        if key not in dic:
            continue
        val = dic[key]
        if 'energy' in key and isinstance(val, list):
            if len(val) == 1:
                val = val[0]
        hess_summary[translate.get(key, key)] = val

    confgen_sub_dir = hess_sub_dir.replace(
        "hessian", "confgen").split("_conf_")[0]
    conf_g = get_conf_g(confgen_sub_dir=confgen_sub_dir)
    update_w_conf_g(conf_g=conf_g,
                    dic=hess_summary)

    return hess_summary


def summarize_endpoints(hess_dir,
                        final_info_dict):

    for i in os.listdir(hess_dir):
        hess_sub_dir = os.path.join(hess_dir, i)
        if not os.path.isdir(hess_sub_dir):
            continue
        end_summary = make_end_summary(hess_sub_dir=hess_sub_dir)
        if end_summary is None:
            continue

        mol_no_h = get_mol(end_summary['smiles'])
        trans_azo = Chem.MolFromSmiles(TRANS_AZO)
        cis_azo = Chem.MolFromSmiles(CIS_AZO)

        is_trans = mol_no_h.HasSubstructMatch(trans_azo,
                                              useChirality=True)
        is_cis = mol_no_h.HasSubstructMatch(cis_azo,
                                            useChirality=True)

        if (not is_trans and not is_cis) or (is_trans and is_cis):
            continue

        new_key = 'trans' if is_trans else 'cis'
        smiles = make_cis(end_summary['smiles'])

        if smiles is None:
            continue

        if smiles not in final_info_dict:
            final_info_dict[smiles] = {}

        sub_dic = final_info_dict[smiles]
        sub_dic[new_key] = end_summary


def make_results_by_mech(final_info_dict):
    for sub_dic in final_info_dict.values():
        ts_lists = sub_dic['transition_states']
        min_g_ts_list = [sorted(i, key=lambda x: x['free_energy'])[0]
                         for i in ts_lists]
        sub_dic['results_by_mechanism'] = {"cis": [], "trans": []}

        for end_key in ['cis', 'trans']:
            for ts_dic in min_g_ts_list:
                ts_summary = {"ts_nxyz": ts_dic["nxyz"],
                              "endpoint_nxyz": sub_dic.get(end_key, {}).get("nxyz")}

                use_keys = ['free_energy', 'energy', 'entropy', 'enthalpy',
                            'free_energy_no_conf', 'conf_free_energy', 'conf_entropy',
                            'vib_entropy']

                end_dic = sub_dic.get(end_key, {})
                for key in use_keys:
                    end_val = end_dic.get(key)
                    if end_val is None:
                        continue
                    delta = ts_dic[key] - end_val
                    if 'energy' in key or 'enthalpy' in key:
                        delta *= const.HARTREE_TO_KCAL_MOL
                    ts_summary["delta_" + key] = delta

                for key in list(ts_summary.keys()):
                    if 'entropy' not in key:
                        continue
                    new_key = key + '_j_mol_k'
                    ts_summary[new_key] = (ts_summary[key] * ENTROPY_CONV)
                    ts_summary.pop(key)

                ts_summary.update({"endpoint_conf_g": end_dic.get("conf_free_energy"),
                                   "ts_conf_g": ts_dic.get("conf_free_energy"),
                                   "mechanism": ts_dic['mechanism']})

                for key in ['endpoint_conf_g', 'ts_conf_g']:
                    if ts_summary.get(key) is not None:
                        ts_summary[key] *= const.AU_TO_KCAL['energy']

                sub_dic['results_by_mechanism'][end_key].append(ts_summary)


def make_summary(final_info_dict):
    for sub_dic in final_info_dict.values():
        if 'summary' not in sub_dic:
            sub_dic['summary'] = {}

        mech_result_dic = sub_dic.get("results_by_mechanism")
        if not mech_result_dic:
            return

        summary = sub_dic['summary']
        for end_key, mech_results in mech_result_dic.items():
            min_g_dic = sorted(mech_results,
                               key=lambda x: x['delta_free_energy'])[0]
            if end_key not in summary:
                summary[end_key] = {}
            summary[end_key]['singlet_barrier'] = min_g_dic


def summarize(base_dir,
              dir_info):
    # ['transition_states', 'product', 'done',
    # 'results_by_mechanism', 'summary']

    # to add to `transition_states`: ['s_t_crossing']

    final_info_dict = {}
    summarize_all_ts(ts_dir=dir_info['evf'],
                     final_info_dict=final_info_dict)
    summarize_endpoints(hess_dir=dir_info['hessian'],
                        final_info_dict=final_info_dict)
    make_results_by_mech(final_info_dict=final_info_dict)
    make_summary(final_info_dict=final_info_dict)

    final_info_dict = filter_by_done(final_info_dict=final_info_dict)

    # `results_by_mechanism['product']`:
    # ['delta_free_energy_s_t_crossing',
    #  's_t_crossing_geom_ids',
    #  'eff_delta_free_energy_s_t_crossing',
    #  's_t_crossing_endpoints',
    #  's_t_crossing_t_isc',
    #  'delta_energy_s_t_crossing',
    #  'delta_entropy_j_mol_k',
    #  'delta_entropy_s_t_crossing',
    #  'delta_enthalpy_s_t_crossing']

    # `summary['product']`: ['s_t_crossing', 'singlet_barrier']

    # `summary['product']['s_t_crossing']:
    # ['delta_free_energy',
    #  'geom_ids',
    #  'eff_delta_free_energy',
    #  'endpoints',
    #  't_isc',
    #  'delta_energy',
    #  'delta_entropy',
    #  'delta_enthalpy',
    #  'mechanism',
    #  'delta_free_energy_no_conf',
    #  'eff_delta_free_energy_no_conf']

    return final_info_dict


def run_all(base_dir):
    info_path = os.path.join(base_dir, 'job_info.json')
    with open(info_path, 'r') as f:
        base_info = json.load(f)

    dir_info = make_all_subdirs(base_dir=base_dir)
    kwargs = {"dir_info": dir_info,
              "base_info": base_info,
              "base_dir": base_dir}

    run_simulations(**kwargs)
    print("Summarizing results...")
    summary = summarize(base_dir=base_dir,
                        dir_info=dir_info)
    save_path = os.path.join(base_dir, 'summary.pickle')
    with open(save_path, 'wb') as f:
        pickle.dump(summary, f)
    print("Summary saved to %s. Workflow complete!" % save_path)


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

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
from ase.build import minimize_rotation_and_translation as align
from ase import Atoms

from nff.utils.confgen import get_mol
from nff.utils.misc import bash_command
from nff.utils import constants as const

from barriers.confgen.neural_confgen import atoms_to_nxyz
from barriers.utils.neuraloptimizer import get_substruc_idx
from barriers.utils.parse import make_isc_props

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

ENTROPY_CONV = 8805.96228743921  # to J / mol K
TEMP = 298.15
KB_HA = 3.167e-6  # atomic units
KB_SI = 1.380649e-23  # SI units
H_PLANCK = 6.62607015e-34  # SI units
MOL = 6.02214076e23
CAL_TO_J = 4.184


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


def add_base(base_info, new_info, key):
    if base_info.get(key) is not None:
        new_info.update(base_info[key])


def make_scan_info_list(rd_dir,
                        base_info):
    info_list = load_rdkit_confgen(rd_dir=rd_dir,
                                   only_trans=True)
    new_info_list = []
    for info in info_list:
        for constraints in MECH_CONSTRAINTS:
            new_info = copy.deepcopy(base_info)
            add_base(base_info, new_info, key='rdkit_confgen')
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
        add_base(base_info, new_info, key='confgen')
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
            add_base(base_info, new_info, key='evf')
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
            add_base(base_info, new_info, key='hessian')
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
                       base_key='irc',
                       mech_key=None):

    folders = [os.path.join(evf_dir, i) for i in os.listdir(evf_dir)]
    confs_per_ts = base_info.get(base_key, {}).get("confs_per_ts")
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
        add_base(base_info, new_info, key=base_key)
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
                                 mech_key='rot',
                                 base_key='triplet_crossing')

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
              description=('conformer generation on the TSs, reactants, '
                           'and products'))


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
    funcs = [
        run_rdkit,
        run_relaxed_scan,
        run_confgen,
        run_hess,
        run_evf,
        run_triplet,
        run_irc
    ]

    for func in funcs:
        func(**kwargs)


def conf_g_from_many(dirs,
                     dic_w_ens):

    ens = []
    for direc in dirs:
        if direc in dic_w_ens:
            these_ens = dic_w_ens[direc]
        else:
            path = os.path.join(direc, 'final_opt.traj')
            trj = Trajectory(path)
            these_ens = np.array([float(i.get_potential_energy())
                                  for i in trj]).tolist()
            dic_w_ens[direc] = these_ens

        ens += these_ens

    ens = np.array(ens)
    kt = KB_HA * TEMP
    rel_ens = (ens - min(ens)) * const.EV_TO_AU
    p = np.exp(-rel_ens / kt)
    p /= p.sum()

    conf_s = KB_HA * (-p * np.log(p)).sum()
    delta_mean_e = (rel_ens * p).sum()
    conf_g = -TEMP * conf_s

    return conf_g, delta_mean_e


def get_confgen_dirs(direc):
    if 'endpoint' in direc:
        return [direc]

    base_dir = "/".join(direc.split("/")[:-1])
    base_name = "_".join(direc.split("_")[:-2])

    all_dirs = []
    for i in os.listdir(base_dir):
        folder = os.path.join(base_dir, i)
        if 'endpoint' in folder:
            continue
        if not os.path.isdir(folder):
            continue
        if base_name not in folder:
            continue
        all_dirs.append(folder)

    return all_dirs


def get_conf_g(confgen_sub_dir,
               dic_w_ens):

    conf_g, de = conf_g_from_many(dirs=[confgen_sub_dir],
                                  dic_w_ens=dic_w_ens)
    confgen_dirs = get_confgen_dirs(direc=confgen_sub_dir)
    total_conf_g, total_de = conf_g_from_many(dirs=confgen_dirs,
                                              dic_w_ens=dic_w_ens)

    results = {"conf_g": conf_g,
               "total_conf_g": total_conf_g,
               "de": de,
               "total_de": total_de}

    return results


def update_w_conf_g(conf_g,
                    de,
                    total_conf_g,
                    total_de,
                    dic):

    free_en = copy.deepcopy(dic['free_energy'])
    enthalpy = copy.deepcopy(dic['enthalpy'])
    vib_entropy = copy.deepcopy(dic['entropy'])

    prefix = 'ts_specific'

    dic.update({"%s_conf_free_energy" % prefix: conf_g,
                "%s_avg_conf_energy" % prefix: de,
                "%s_conf_entropy" % prefix: -conf_g,
                "%s_entropy" % prefix: vib_entropy - conf_g,
                "%s_free_energy" % prefix: free_en + conf_g + de,
                "%s_enthalpy" % prefix: enthalpy + de})

    dic.update({"conf_free_energy": total_conf_g,
                "avg_conf_energy": total_de,
                "conf_entropy": - total_conf_g,
                "vib_entropy": vib_entropy,
                "entropy": vib_entropy - total_conf_g,
                "free_energy": free_en + total_conf_g + total_de,
                "enthalpy": enthalpy + total_de})


def make_ts_summary(ts_sub_dir,
                    dic_w_ens):

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
                  "mechanism": old_info["mechanism"],
                  "confnum": old_info["confnum"]}

    for key in ts_keys:
        if key not in dic:
            continue
        ts_summary[translate.get(key, key)] = dic[key]

    # add conformational free energy
    confgen_sub_dir = ts_sub_dir.replace("evf", "confgen").split("_conf_")[0]
    conf_results = get_conf_g(confgen_sub_dir=confgen_sub_dir,
                              dic_w_ens=dic_w_ens)

    update_w_conf_g(conf_g=conf_results["conf_g"],
                    de=conf_results["de"],
                    total_conf_g=conf_results["total_conf_g"],
                    total_de=conf_results["total_de"],
                    dic=ts_summary)

    return ts_summary


def make_general_smiles(smiles,
                        stereo):
    substruc_idx = get_substruc_idx(template_smiles=TRANS_AZO,
                                    smiles=smiles)
    nn_idx = substruc_idx[4:6]
    new_mol = Chem.MolFromSmiles(smiles)
    nn_bond_pairs = [[i, b] for i, b in enumerate(new_mol.GetBonds()) if
                     bond_idx_match(bond=b, idx=nn_idx)]
    if len(nn_bond_pairs) != 1:
        print(("Problem finding the N=N bond in the azobenzene substructure "
               "of smiles %s. Skipping" % smiles))
        return

    bond_idx, bond = nn_bond_pairs[0]
    bond.SetStereo(stereo)

    new_smiles = Chem.MolToSmiles(
        Chem.MolFromSmiles(
            Chem.MolToSmiles(
                new_mol
            )
        )
    )

    return new_smiles


def make_cis(smiles):
    return make_general_smiles(smiles=smiles,
                               stereo=Chem.BondStereo.STEREOZ)


def make_trans(smiles):
    return make_general_smiles(smiles=smiles,
                               stereo=Chem.BondStereo.STEREOE)


def summarize_all_ts(ts_dir,
                     final_info_dict):

    dic_w_ens = {}

    for i in os.listdir(ts_dir):
        ts_sub_dir = os.path.join(ts_dir, i)
        if not os.path.isdir(ts_sub_dir):
            continue
        ts_summary = make_ts_summary(ts_sub_dir=ts_sub_dir,
                                     dic_w_ens=dic_w_ens)
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

    return dic_w_ens


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


def make_end_summary(hess_sub_dir,
                     dic_w_ens):

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
    conf_results = get_conf_g(confgen_sub_dir=confgen_sub_dir,
                              dic_w_ens=dic_w_ens)
    update_w_conf_g(conf_g=conf_results["conf_g"],
                    de=conf_results["de"],
                    total_conf_g=conf_results["total_conf_g"],
                    total_de=conf_results["total_de"],
                    dic=hess_summary)

    return hess_summary


def summarize_endpoints(hess_dir,
                        final_info_dict,
                        dic_w_ens):

    for i in os.listdir(hess_dir):
        hess_sub_dir = os.path.join(hess_dir, i)
        if not os.path.isdir(hess_sub_dir):
            continue
        end_summary = make_end_summary(hess_sub_dir=hess_sub_dir,
                                       dic_w_ens=dic_w_ens)
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


def make_mech_ts_summary(ts_dic,
                         sub_dic,
                         end_key,
                         trace=False):

    ts_summary = {"ts_nxyz": ts_dic["nxyz"],
                  "endpoint_nxyz": sub_dic.get(end_key, {}).get("nxyz")}

    use_keys = ['free_energy',
                'energy',
                'entropy',
                'enthalpy',
                'free_energy_no_conf',
                'conf_free_energy',
                'conf_entropy',
                'vib_entropy']

    use_keys += ["eff_%s" % i for i in use_keys]

    end_dic = sub_dic.get(end_key, {})
    for key in use_keys:
        # if key == 'eff_free_energy':
        # import pdb
        # pdb.set_trace()
        end_val = end_dic.get(key)
        if end_val is None:
            end_val = end_dic.get(key.replace("eff_", ""))
        if end_val is None:
            continue

        if key not in ts_dic:
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

    ts_summary.update({"endpoint_conf_free_energy": end_dic.get("conf_free_energy"),
                       "ts_conf_free_energy": ts_dic.get("conf_free_energy")})
    for key in ['mechanism', 'confnum']:
        if ts_dic.get(key) is not None:
            ts_summary.update({key: ts_dic[key]})

    # for key in ['endpoint_conf_free_energy', 'ts_conf_free_energy']:
    #     if ts_summary.get(key) is not None:
    #         ts_summary[key] *= const.AU_TO_KCAL['energy']

    return ts_summary


def update_mech_dic(ts_dic,
                    sub_dic,
                    end_key):

    ts_summary = make_mech_ts_summary(ts_dic=ts_dic,
                                      sub_dic=sub_dic,
                                      end_key=end_key)
    sub_dic['results_by_mechanism'][end_key]['ts'].append(ts_summary)


def make_results_by_mech(final_info_dict):
    for sub_dic in final_info_dict.values():
        ts_lists = sub_dic['transition_states']
        min_g_ts_list = [sorted(i, key=lambda x: x['free_energy'])[0]
                         for i in ts_lists]
        sub_dic['results_by_mechanism'] = {key: {"s_t_crossing": [],
                                                 "ts": []} for key in ['cis', 'trans']}

        for end_key in ['cis', 'trans']:
            for ts_dic in min_g_ts_list:
                update_mech_dic(ts_dic=ts_dic,
                                sub_dic=sub_dic,
                                end_key=end_key)


def select_isc(ts_list):
    dics = [i for i in ts_list if 's_t_crossing' in i]
    return dics


def update_mech_w_isc(final_info_dict):
    for sub_dic in final_info_dict.values():
        ts_lists = [select_isc(i) for i in sub_dic['transition_states']]
        sort_g_dics = [sorted(ts_list, key=lambda x: x.get("eff_free_energy",
                                                           float("inf")))
                       for ts_list in ts_lists]
        min_g_dics = [dics[0] for dics in sort_g_dics if dics]

        mech_results = sub_dic['results_by_mechanism']

        for dic in min_g_dics:
            for end_key in ['cis', 'trans']:
                s_t_dics = dic['s_t_crossing']
                s_t_summaries = []
                for s_t_dic in s_t_dics:
                    if not s_t_dic['converged']:
                        continue
                    s_t_summary = make_mech_ts_summary(ts_dic=s_t_dic,
                                                       sub_dic=sub_dic,
                                                       end_key=end_key,
                                                       trace=True)
                    s_t_summary.update({"endpoint": s_t_dic["endpoint"]})
                    s_t_summaries.append(s_t_summary)

                mech_results[end_key]['s_t_crossing'].append(s_t_summaries)


def get_effective_isc_dg(props):

    k_isc = 1 / props['t_isc']
    # delta S from k_isc, in J/(mol K)
    ds = KB_SI * (np.log(H_PLANCK * k_isc / (KB_SI * TEMP)) - 1 / 2) * MOL

    # corresponding dg, in Ha
    dg = -ds * TEMP / 1000 / CAL_TO_J * const.KCAL_TO_AU['energy']

    return dg


def nxyz_to_atoms(nxyz):
    atoms = Atoms(numbers=np.array(nxyz)[:, 0],
                  positions=np.array(nxyz)[:, 1:])
    return atoms


def determine_triplet_side(props_list,
                           react_idx,
                           cis_nxyz=None,
                           trans_nxyz=None):
    """
    From two positions of singlet/triplet crossings, figure out which is closer to the
    cis and which closer to trans.
    """

    msg = "Need to provide either the reactant or product nxyz"
    assert (trans_nxyz is not None) or (cis_nxyz is not None), msg
    atoms_dic = {"trans": trans_nxyz, "cis": cis_nxyz}
    use_dic = {key: val for key, val in atoms_dic.items() if val is not None}

    key = list(use_dic.keys())[0]
    eff_ref_atoms = nxyz_to_atoms(np.array(use_dic[key])[react_idx])
    rmsds = []

    for props in props_list:
        crossing_nxyz = props['nxyz']
        eff_cross_atoms = nxyz_to_atoms(np.array(crossing_nxyz)[react_idx])
        align(eff_ref_atoms, eff_cross_atoms)

        num_atoms = len(eff_ref_atoms)
        rmsd = (((eff_cross_atoms.get_positions() -
                  eff_ref_atoms.get_positions()) ** 2).sum() / num_atoms) ** 0.5

        rmsds.append(rmsd)

    argmin = int(np.argmin(rmsds))
    other_key = "trans" if key == "cis" else "cis"
    other_arg = 1 - argmin
    summary_dic = {argmin: key, other_arg: other_key}
    summary_list = [summary_dic[i] for i in range(2)]

    for i, endpoint in enumerate(summary_list):
        props_list[i].update({"endpoint": endpoint})


def get_some_isc_info(final_info_dict,
                      cis_smiles):
    cis_nxyz = final_info_dict.get(cis_smiles, {}).get("cis", {}).get("nxyz")
    trans_nxyz = final_info_dict.get(
        cis_smiles, {}).get("trans", {}).get("nxyz")

    if cis_nxyz is None and trans_nxyz is None:
        return

    substruc_idx = get_substruc_idx(template_smiles=TRANS_AZO,
                                    smiles=cis_smiles)
    react_idx = substruc_idx[4:6]

    return react_idx, cis_nxyz, trans_nxyz


def fix_t_isc(props_list,
              final_info_dict,
              cis_smiles):

    sub_dic = final_info_dict[cis_smiles]

    for props in props_list:
        endpoint = props['endpoint']
        endpoint_g = sub_dic[endpoint]['free_energy']
        other_endpoint = 'cis' if endpoint == 'trans' else 'cis'
        other_g = sub_dic[other_endpoint]['free_energy']

        # if you're on the lower energy side, multiply k by 3 (because you're going
        # from 3 triplets to one singlet)

        if endpoint_g < other_g:
            factor = 3
        else:
            factor = 2

        if props.get('k_isc') is not None:
            props['k_isc'] *= factor
        if props.get('t_isc') is not None:
            props['t_isc'] /= factor


def make_isc_summary(isc_sub_dir,
                     ts_dic,
                     final_info_dict,
                     cis_smiles):

    isc_path = os.path.join(isc_sub_dir, 'triplet_opt.pickle')
    if not os.path.isfile(isc_path):
        return

    info_path = os.path.join(isc_sub_dir, 'job_info.json')
    if not os.path.isfile(info_path):
        return

    props_list = make_isc_props(isc_sub_dir)

    for props in props_list:
        if 's_t_gap' in props:
            props.update({"s_t_gap_kcal": (props['s_t_gap'] *
                                           const.AU_TO_KCAL['energy'])})

        if not props['converged']:
            continue

        hess_keys = ['freeenergy', 'enthalpy', 'entropy']
        if all([key in props for key in hess_keys]):
            props['free_energy'] = props['freeenergy']
            props.pop('freeenergy')

        else:
            # if a Hessian calculation wasn't done in the isc simulation, then inherit
            # the rovibrational quantities from the TS

            props['energy'] = props['singlet_energy']
            base_keys = ['entropy',
                         'free_energy',
                         'enthalpy',
                         'conf_entropy',
                         'conf_free_energy']

            for ts_key, ts_val in ts_dic.items():
                if ts_key not in base_keys:
                    continue
                if ts_key in ['entropy', 'conf_entropy', 'conf_free_energy']:
                    props[ts_key] = ts_val
                else:
                    delta = ts_val - ts_dic['energy']
                    props[ts_key] = props['energy'] + delta

    out = get_some_isc_info(final_info_dict=final_info_dict,
                            cis_smiles=cis_smiles)
    if out is None:
        return props_list

    react_idx, cis_nxyz, trans_nxyz = out
    determine_triplet_side(props_list=props_list,
                           react_idx=react_idx,
                           cis_nxyz=cis_nxyz,
                           trans_nxyz=trans_nxyz)

    # divide by 2 or 3 depending on which side you're on
    fix_t_isc(props_list=props_list,
              final_info_dict=final_info_dict,
              cis_smiles=cis_smiles)

    for props in props_list:
        if not props['converged']:
            continue
        isc_dg = get_effective_isc_dg(props=props)
        props.update({"eff_free_energy": props['free_energy'] + isc_dg,
                      "eff_entropy": props['entropy'] - isc_dg})
    return props_list


def summarize_all_isc(isc_dir,
                      final_info_dict):

    for i in os.listdir(isc_dir):
        isc_sub_dir = os.path.join(isc_dir, i)
        if not os.path.isdir(isc_sub_dir):
            continue

        info_path = os.path.join(isc_sub_dir, 'job_info.json')
        with open(info_path, 'r') as f:
            info = json.load(f)

        cis_smiles = make_cis(info['smiles'])
        mech = info['mechanism']
        confnum = info['confnum']

        ts_dics = []
        for ts_list in final_info_dict[cis_smiles]['transition_states']:
            for ts in ts_list:
                this_mech = ts['mechanism']
                this_confum = ts['confnum']

                if this_mech == mech and this_confum == confnum:
                    ts_dics.append(ts)

        assert len(ts_dics) == 1, "Something has gone wrong"
        ts_dic = ts_dics[0]
        isc_summary = make_isc_summary(isc_sub_dir=isc_sub_dir,
                                       ts_dic=ts_dic,
                                       final_info_dict=final_info_dict,
                                       cis_smiles=cis_smiles)
        if 's_t_crossing' not in ts_dic:
            ts_dic['s_t_crossing'] = []
        ts_dic['s_t_crossing'] += isc_summary


def get_min_st_dics(mech_results_dic,
                    sub_dic):

    s_t_crossing_dics = mech_results_dic.get('s_t_crossing')
    if not s_t_crossing_dics:
        return
    use_dics = [i for i in s_t_crossing_dics if i]
    if not use_dics:
        return

    eff_g_list = []
    for dic_list in use_dics:
        end_keys = [dic['endpoint'] for dic in dic_list]
        endpoint_g_list = [sub_dic[end_key]['free_energy'] for
                           end_key in end_keys]

        # eff_g on the side closer to the reactant
        idx = np.argmin(endpoint_g_list)
        eff_g = dic_list[idx]['delta_eff_free_energy']
        eff_g_list.append(eff_g)

    argmin = np.argmin(eff_g_list)
    min_g_dics = use_dics[argmin]

    return min_g_dics


def add_extra_info(summary,
                   smiles,
                   sub_dic):

    cis_smiles = make_cis(smiles=smiles)
    trans_smiles = make_trans(smiles=smiles)

    sort_keys = sorted(['cis', 'trans'],
                       key=lambda x: sub_dic[x]['free_energy'])
    stable = sort_keys[0]
    unstable = sort_keys[1]

    sub_dic.update({"cis_smiles": cis_smiles,
                    "trans_smiles": trans_smiles,
                    "stable": stable,
                    "unstable": unstable})


def update_w_st(summary,
                end_key,
                min_st_dics):

    import pdb
    pdb.set_trace()

    if 's_t_crossing' not in summary:
        summary['s_t_crossing'] = {}
    summary[end_key]['s_t_crossing'] = min_st_dics


def make_summary(final_info_dict):
    for smiles, sub_dic in final_info_dict.items():
        if 'summary' not in sub_dic:
            sub_dic['summary'] = {}

        mech_result_dic = sub_dic.get("results_by_mechanism")
        if not mech_result_dic:
            return

        summary = sub_dic['summary']
        for end_key, mech_results_dic in mech_result_dic.items():
            mech_results = mech_results_dic['ts']
            min_g_dic = sorted(mech_results,
                               key=lambda x: x['delta_free_energy'])[0]

            summary[end_key] = min_g_dic

            min_st_dics = get_min_st_dics(mech_results_dic=mech_results_dic,
                                          sub_dic=sub_dic)

            update_w_st(summary=summary,
                        end_key=end_key,
                        min_st_dics=min_st_dics)

        add_extra_info(summary=summary,
                       sub_dic=sub_dic,
                       smiles=smiles)


def summarize(base_dir,
              dir_info):

    final_info_dict = {}
    dic_w_ens = summarize_all_ts(ts_dir=dir_info['evf'],
                                 final_info_dict=final_info_dict)

    # needs to be before isc so we can figure out which side each one is on
    summarize_endpoints(hess_dir=dir_info['hessian'],
                        final_info_dict=final_info_dict,
                        dic_w_ens=dic_w_ens)
    summarize_all_isc(isc_dir=dir_info['triplet_crossing'],
                      final_info_dict=final_info_dict)

    make_results_by_mech(final_info_dict=final_info_dict)
    update_mech_w_isc(final_info_dict=final_info_dict)

    make_summary(final_info_dict=final_info_dict)

    final_info_dict = filter_by_done(final_info_dict=final_info_dict)

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

import os
import pickle
import numpy as np

from barriers.irc.neural_irc import COMPLETION_MESSAGE as IRC_COMPLETION_MSG

TRIPLET_LOG = 'neural_triplet_crossing.log'

HESS_PARSE_KEYS = ["vibdisps",
                   "vibfreqs",
                   "hessianmatrix",
                   "imgfreq",
                   "freeenergy",
                   "enthalpy",
                   "entropy",
                   "energy"]


def load_isc(job_dir):

    save_path = os.path.join(job_dir, 'triplet_opt.pickle')
    with open(save_path, 'rb') as f:
        summary = pickle.load(f)

    k_isc = summary['opt']['k_isc']
    mode_results = summary['opt']['mode_results']

    sub_info = summary['opt']
    trj = sub_info['atoms']
    singlet_ens = sub_info['singlet_energies']
    triplet_ens = sub_info['triplet_energies']
    nxyz_list = [np.concatenate([atoms.get_atomic_numbers().reshape(-1, 1),
                                 atoms.get_positions()], axis=-1).tolist()
                 for atoms in trj]

    return nxyz_list, singlet_ens, triplet_ens, mode_results, k_isc


def update_w_hess(mode_dic,
                  props):

    if mode_dic is None:
        return
    hess_quants = {key: mode_dic[key] for key in HESS_PARSE_KEYS
                   if key in mode_dic}
    props.update(hess_quants)


def is_converged(job_dir):
    out_path = os.path.join(job_dir, TRIPLET_LOG)
    with open(out_path, 'r') as f:
        lines = f.readlines()

    assert (IRC_COMPLETION_MSG + "\n") in reversed(lines)

    converged = []
    for line in lines:
        if 'Found a triplet crossing' in line:
            converged.append(True)
        if 'Did not find a triplet crossing ' in line:
            converged.append(False)

    # if there's 4 of them, it's because a second run was done for
    # each with smaller step sizes to get a more accurate crossing. Therefore
    # in that case you only want to get the second 'converged' sign

    if len(converged) == 4:
        converged = [converged[1], converged[3]]

    return converged


def make_isc_props(job_dir):

    out = load_isc(job_dir=job_dir)
    nxyz_list, singlet_ens, triplet_ens, mode_results, k_isc_list = out
    all_conv = is_converged(job_dir)

    props_list = []

    for i, nxyz in enumerate(nxyz_list):
        singlet_en = singlet_ens[i]
        triplet_en = triplet_ens[i]
        k_isc = k_isc_list[i]
        converged = all_conv[i]

        props = {"singlet_energy": singlet_en,
                 "triplet_energy": triplet_en,
                 "s_t_gap": (singlet_en - triplet_en),
                 "nxyz": nxyz,
                 "k_isc": k_isc,
                 "converged": converged}
        if k_isc is not None:
            props["t_isc"] = 1 / k_isc
        update_w_hess(mode_dic=mode_results[i],
                      props=props)
        props_list.append(props)

    return props_list

"""
Script for optimizing a transition state from an initial guess
using eigenvector following.

"""


import argparse
import numpy as np
import pickle

from nff.md.tully.io import coords_to_xyz
from nff.reactive_tools.ev_following import ev_run
from nff.io.ase import AtomsBatch
from nff.utils.constants import EV_TO_AU, BOHR_RADIUS

from barriers.utils.ase_neb import get_calc_kwargs, load_params
from barriers.utils.vib import hessian_and_modes


COMPLETION_MESSAGE = "Eigenvector following terminated normally."
SAVE_PATH = 'ts.json'
PICKLE_SAVE_PATH = 'ts.pickle'
DEFAULT_THERMO_KWARGS = {"flip_all_but_ts": True,
                         "imag_cutoff": 0}


def get_atoms(coords,
              atoms_kwargs,
              nxyz=None):
    if nxyz is None and coords is not None:
        nxyz = coords_to_xyz(coords=coords).astype('float')
    elif nxyz is not None:
        nxyz = np.array(nxyz)
    else:
        msg = "Must specify either nxyz or coords"
        raise Exception(msg)

    trimmed_kwargs = {key: val for key, val in atoms_kwargs.items()
                      if key != 'nbr_update_period'}
    atoms_batch = AtomsBatch(numbers=nxyz[:, 0],
                             positions=nxyz[:, 1:],
                             **trimmed_kwargs)

    return atoms_batch


def to_json(dic):
    for key, val in dic.items():
        if hasattr(val, "tolist"):
            dic[key] = val.tolist()
    return dic


def add_converged(atoms,
                  results,
                  ts_cutoff,
                  mode_results,
                  imag_cutoff,
                  ev_kwargs):

    freqs = np.array(mode_results['vibfreqs'])
    imgfreq = len((freqs < imag_cutoff).nonzero()[0])
    grad = atoms.get_forces()

    converged = fmax < ev_kwargs["convergence"]
    converged = all([converged,
                     freqs[0] <= ts_cutoff,
                     imgfreq == 1])

    results['converged'] = converged


def save_results(xyz,
                 grad,
                 energy,
                 rmslist,
                 maxlist,
                 atoms,
                 mode_results,
                 imag_cutoff,
                 ts_cutoff,
                 ev_kwargs):

    numbers = atoms.get_atomic_numbers()
    nxyz = np.concatenate([numbers.reshape(-1, 1),
                           xyz.detach().cpu().numpy().reshape(-1, 3)], axis=-1)

    grad_conv = EV_TO_AU * BOHR_RADIUS
    forces = (-grad.reshape(-1, 3) * grad_conv)

    # don't save everything, like the mass-weighted and mass-weighted projected
    # Hessians, as they take up a lot of space and cause parsing to really slow
    # down

    exclude_keys = ["hess_proj", "mwhess_proj"]
    add_results = {key: val for key, val in to_json(mode_results).items()
                   if key not in exclude_keys}

    add_converged(atoms=atoms,
                  results=add_results,
                  ts_cutoff=ts_cutoff,
                  mode_results=mode_results,
                  imag_cutoff=imag_cutoff,
                  ev_kwargs=ev_kwargs)

    info = {"nxyz": nxyz.tolist(),
            "forces": forces.tolist(),
            "energy": energy * EV_TO_AU,
            "rms_grad": rmslist[-1].item() * grad_conv,
            "max_grad": maxlist[-1].item() * grad_conv}
    info.update(add_results)

    with open(PICKLE_SAVE_PATH, 'wb') as f:
        pickle.dump(info, f)


def run_from_atoms(atoms,
                   calc_kwargs,
                   ev_kwargs,
                   atoms_kwargs,
                   thermo_kwargs,
                   ts_cutoff):

    nbr_update_period = atoms_kwargs["nbr_update_period"]
    output = ev_run(ev_atoms=atoms,
                    calc_kwargs=calc_kwargs,
                    nbr_update_period=nbr_update_period,
                    **ev_kwargs)
    print(COMPLETION_MESSAGE)

    xyz, grad, _, rmslist, maxlist = output
    # This should work now because the same atoms we inputted
    # are the ones we get out, just with different positions
    energy = atoms.get_potential_energy().item()

    # Entropy, enthalpy, etc.

    flip_all_but_ts = thermo_kwargs.get("flip_all_but_ts", True)
    imag_cutoff = thermo_kwargs.get("imag_cutoff")

    mode_results = hessian_and_modes(atoms,
                                     flip_all_but_ts=flip_all_but_ts,
                                     imag_cutoff=imag_cutoff)

    save_results(xyz=xyz,
                 grad=grad,
                 energy=energy,
                 rmslist=rmslist,
                 maxlist=maxlist,
                 atoms=atoms,
                 mode_results=mode_results,
                 imag_cutoff=imag_cutoff,
                 ts_cutoff=ts_cutoff,
                 ev_kwargs=ev_kwargs)

    return output


def run_from_params(params):
    calc_kwargs = get_calc_kwargs(params)
    ev_kwargs = params["ev_kwargs"]
    atoms_kwargs = params["atoms_kwargs"]
    thermo_kwargs = params.get("thermo_kwargs",
                               DEFAULT_THERMO_KWARGS)

    atoms = get_atoms(coords=params.get("coords"),
                      atoms_kwargs=atoms_kwargs,
                      nxyz=params.get("nxyz"))

    output = run_from_atoms(atoms=atoms,
                            calc_kwargs=calc_kwargs,
                            ev_kwargs=ev_kwargs,
                            atoms_kwargs=atoms_kwargs,
                            thermo_kwargs=thermo_kwargs,
                            ts_cutoff=params["cutoff_to_be_ts"])

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--info_file', type=str,
                        help=('The path with to the config file'),
                        default='job_info.json')
    args = parser.parse_args()

    params = load_params(file=args.info_file)
    run_from_params(params)


if __name__ == "__main__":
    main()

from neuralnet.vib import hessian_and_modes
from neuralnet.utils.barriers.evf import get_atoms, to_json
import json
import argparse
from rdkit import Chem
from barriers.utils.ase_neb import get_calc_kwargs, load_params
from nff.io.ase import NeuralFF
from nff.utils import constants as const

PERIODICTABLE = Chem.GetPeriodicTable()
HESS_FILENAME = "hess.json"
DEFAULT_INFO_FILE = "job_info.json"
COORDS_NAME = "coords.json"


def get_modes(params):

    atoms = get_atoms(coords=params.get("coords"),
                      atoms_kwargs=params["atoms_kwargs"],
                      nxyz=params.get('nxyz'))

    calc_kwargs = get_calc_kwargs(params=params)
    nff = NeuralFF.from_file(**calc_kwargs)
    atoms.set_calculator(nff)

    results = {
        "energy": atoms.get_potential_energy() * const.EV_TO_AU,
        "forces": atoms.get_forces() * const.EV_TO_AU * const.KCAL_TO_AU['_grad']
    }
    mode_results = hessian_and_modes(atoms,
                                     imag_cutoff=params['imag_cutoff'],
                                     rotor_cutoff=params['rotor_cutoff'],
                                     temperature=params['temperature'],
                                     pressure=params['pressure'],
                                     flip_all_but_ts=False,
                                     analytical=params['analytical_hessian'])
    results.update(mode_results)
    results = to_json(results)

    return results


def main(params):
    mode_results = get_modes(params)

    # save in a JSON
    with open(HESS_FILENAME, "w") as f:
        json.dump(mode_results, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="run a neural Hessian calculation")
    parser.add_argument('--info_file', type=str, default=DEFAULT_INFO_FILE,
                        help="file containing all parameters")

    args = parser.parse_args()
    params = load_params(file=args.info_file)
    main(params)

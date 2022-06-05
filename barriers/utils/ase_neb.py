"""
Script for using NEB to get barriers using ASE. More recent than the scripts in
`djangochem/neuralnet` so better suited for changes to NFF models.

Note: Requires ASE version 3.22.0

To get the starting coords, do the following:

```
images = rxn_path.images.filter(converged=True)
path_images = (rxn_path.pathimage_set.filter(geometry__converged=True)
               .order_by("imagenumber"))
coord_set = [i.geometry.get_coords() for i in path_images]
```


"""

import os
import json
import numpy as np
import torch
import argparse

from ase import Atoms, optimize
from ase.io import Trajectory
from ase.neb import NEB, BaseNEB, NEBState, DyNEB
from ase.build import minimize_rotation_and_translation
from ase.optimize.precon import Precon, PreconImages


from nff.md.tully.io import coords_to_xyz
from nff.io.ase import AtomsBatch, NeuralFF

NEB_TRJ = "neb.traj"
NEB_LOG = "neb.log"

OPT_TRJ = "opt_{num}.traj"
OPT_LOG = "opt_{num}.log"

FINAL_TRJ = "band_{num}.traj"

TERMINATION_LINE = "Neural NEB terminated normally.\n"
CONVERGED_LINE = "Neural NEB converged.\n"
NON_CONVERGED_LINE = "Neural NEB did not converge.\n"


class NeuralBaseNEB(BaseNEB):
    def __init__(self,
                 *args,
                 atoms_kwargs,
                 calc_kwargs,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.calculator = NeuralFF.from_file(**calc_kwargs)
        self.atoms_kwargs = atoms_kwargs
        self.num_atoms = torch.LongTensor([len(i) for i in
                                           self.images[1:-1]])

        self.nbr_update_period = atoms_kwargs["nbr_update_period"]
        self.nsteps = 0

    @property
    def atoms_batch(self):
        intermed_images = self.images[1:-1]
        _atoms_batch = make_atoms_batch(calculator=self.calculator,
                                        atoms_kwargs=self.atoms_kwargs,
                                        images=intermed_images,
                                        num_atoms=self.num_atoms)

        return _atoms_batch

    def get_forces(self):
        """
        Copied from BaseNEB. The only change is in how we compute the forces
        and the tracking of number of steps

        Evaluate and return the forces."""
        images = self.images
        forces = np.empty(((self.nimages - 2), self.natoms, 3))
        energies = np.empty(self.nimages)

        if self.remove_rotation_and_translation:
            for i in range(1, self.nimages):
                minimize_rotation_and_translation(images[i - 1], images[i])

        if self.method != 'aseneb':
            energies[0] = images[0].get_potential_energy()
            energies[-1] = images[-1].get_potential_energy()

        if self.nsteps % self.nbr_update_period == 0:
            self.atoms_batch.update_nbr_list()

        these_ens, forces = get_ens_forces(atoms_batch=self.atoms_batch,
                                           num_atoms=self.num_atoms)
        energies[1:-1] = these_ens
        self.nsteps += 1

        # if this is the first force call, we need to build the preconditioners
        if (self.precon is None or isinstance(self.precon, str) or
                isinstance(self.precon, Precon)):
            self.precon = PreconImages(self.precon, images)

        # apply preconditioners to transform forces
        # for the default IdentityPrecon this does not change their values
        precon_forces = self.precon.apply(forces, index=slice(1, -1))

        # Save for later use in iterimages:
        self.energies = energies
        self.real_forces = np.zeros((self.nimages, self.natoms, 3))
        self.real_forces[1:-1] = forces

        state = NEBState(self, images, energies)

        # Can we get rid of self.energies, self.imax, self.emax etc.?
        self.imax = state.imax
        self.emax = state.emax

        spring1 = state.spring(0)

        self.residuals = []
        for i in range(1, self.nimages - 1):
            spring2 = state.spring(i)
            tangent = self.neb_method.get_tangent(state, spring1, spring2, i)

            # Get overlap between full PES-derived force and tangent
            tangential_force = np.vdot(forces[i - 1], tangent)

            # from now on we use the preconditioned forces (equal for precon=ID)
            imgforce = precon_forces[i - 1]

            if i == self.imax and self.climb:
                """The climbing image, imax, is not affected by the spring
                   forces. This image feels the full PES-derived force,
                   but the tangential component is inverted:
                   see Eq. 5 in paper II."""
                if self.method == 'aseneb':
                    tangent_mag = np.vdot(tangent, tangent)  # For normalizing
                    imgforce -= 2 * tangential_force / tangent_mag * tangent
                else:
                    imgforce -= 2 * tangential_force * tangent
            else:
                self.neb_method.add_image_force(state, tangential_force,
                                                tangent, imgforce, spring1,
                                                spring2, i)
                # compute the residual - with ID precon, this is just max force
                residual = self.precon.get_residual(i, imgforce)
                self.residuals.append(residual)

            spring1 = spring2

        return precon_forces.reshape((-1, 3))


class NeuralDyNEB(NeuralBaseNEB, DyNEB):
    def __init__(self,
                 *args,
                 atoms_kwargs,
                 calc_kwargs,
                 **kwargs):
        DyNEB.__init__(self, *args, **kwargs)
        NeuralBaseNEB.__init__(self, *args,
                               atoms_kwargs=atoms_kwargs,
                               calc_kwargs=calc_kwargs,
                               **kwargs)

    def get_forces(self):
        forces = super().get_forces()
        if not self.dynamic_relaxation:
            return forces

        """Get NEB forces and scale the convergence criteria to focus
           optimization on saddle point region. The keyword scale_fmax
           determines the rate of convergence scaling."""
        n = self.natoms
        for i in range(self.nimages - 2):
            n1 = n * i
            n2 = n1 + n
            force = np.sqrt((forces[n1:n2] ** 2.).sum(axis=1)).max()
            n_imax = (self.imax - 1) * n  # Image with highest energy.

            positions = self.get_positions()
            pos_imax = positions[n_imax:n_imax + n]

            """Scale convergence criteria based on distance between an
               image and the image with the highest potential energy."""
            rel_pos = np.sqrt(((positions[n1:n2] - pos_imax) ** 2).sum())
            if force < self.fmax * (1 + rel_pos * self.scale_fmax):
                if i == self.imax - 1:
                    # Keep forces at saddle point for the log file.
                    pass
                else:
                    # Set forces to zero before they are sent to optimizer.
                    forces[n1:n2, :] = 0
        return forces


class NeuralNEB(NeuralDyNEB, NEB):
    def __init__(self,
                 *args,
                 atoms_kwargs,
                 calc_kwargs,
                 **kwargs):

        NEB.__init__(self, *args, **kwargs)
        NeuralDyNEB.__init__(self, *args,
                             atoms_kwargs=atoms_kwargs,
                             calc_kwargs=calc_kwargs,
                             **kwargs)


def make_atoms_batch(calculator,
                     atoms_kwargs,
                     images,
                     num_atoms):

    numbers = np.concatenate([i.get_atomic_numbers()
                              for i in images])
    positions = np.concatenate([i.get_positions()
                                for i in images])
    cells = np.array([i.get_cell()
                        for i in images])

    trimmed_kwargs = {key: val for key, val in atoms_kwargs.items()
                      if key != 'nbr_update_period'}
    _atoms_batch = AtomsBatch(numbers=numbers,
                              positions=positions,
                              cell=cells[0],
                              pbc=cells[0] is not None,
                              props={"num_atoms": num_atoms},
                              **trimmed_kwargs)

    _atoms_batch.set_calculator(calculator)

    return _atoms_batch


def get_ens_forces(atoms_batch,
                   num_atoms):

    energies = atoms_batch.get_potential_energy()
    cat_forces = atoms_batch.get_forces()
    forces = torch.stack(
        torch.split(
            torch.Tensor(cat_forces), num_atoms.tolist()
        )
    ).numpy()

    return energies, forces


def get_optim(atoms,
              optim_name,
              optim_kwargs,
              logfile,
              trj):

    optim_class = getattr(optimize, optim_name)
    optim_kwargs.update({"logfile": logfile,
                         "trajectory": trj})
    optimizer = optim_class(atoms, **optim_kwargs)

    return optimizer


def opt_start_end(images,
                  neb_params,
                  calc_kwargs,
                  atoms_kwargs):
    """
    Optimize the first and last images 

    """

    from ase.io.vasp import write_vasp

    print("Doing initial optimization")
    opt_images = [images[0], images[-1]]
    calculator = NeuralFF.from_file(**calc_kwargs)

    for i, image in enumerate(opt_images):

        trimmed_kwargs = {key: val for key, val in atoms_kwargs.items()
                          if key != 'nbr_update_period'}
        atoms_batch = AtomsBatch(numbers=image.get_atomic_numbers(),
                                 positions=image.get_positions(),
                                 cell=image.get_cell(),
                                 pbc=image.get_pbc(),
                                 **trimmed_kwargs)
        atoms_batch.update_nbr_list()
        atoms_batch.set_calculator(calculator)

        optimizer = get_optim(atoms=atoms_batch,
                              optim_name=neb_params["init_optim_name"],
                              optim_kwargs=neb_params["init_optim_kwargs"],
                              logfile=OPT_LOG.format(num=i),
                              trj=OPT_TRJ.format(num=i))
        optimizer.run(**neb_params["optim_run_kwargs"])

        new_image = Atoms(numbers=atoms_batch.get_atomic_numbers(),
                          positions=atoms_batch.get_positions(),
                          cell=atoms_batch.get_cell(),
                          pbc=atoms_batch.get_pbc())
        opt_images[i] = new_image
        print('optimized energy is ')
        new_image.set_calculator(calculator)
        print(new_image.get_potential_energy())

    images[0] = opt_images[0]
    images[-1] = opt_images[-1]


def get_neb(images,
            neb_params,
            atoms_kwargs,
            calc_kwargs):

    neb_kwargs = neb_params["neb_kwargs"]
    neb = NeuralNEB(images,
                    atoms_kwargs=atoms_kwargs,
                    calc_kwargs=calc_kwargs,
                    **neb_kwargs)

    return neb


def get_images(params):

    if params["neb_params"]["interpolation_needed"]:
        reactantcoords = params['reactantcoords']
        productcoords = params['productcoords']
        reactantnxyz = coords_to_xyz(coords=reactantcoords).astype('float')
        productnxyz = coords_to_xyz(coords=productcoords).astype('float')
        if 'reactantlattice' in params:
            reactantlattice = params['reactantlattice']
            productlattice = params['productlattice']
            reactant = Atoms(reactantnxyz[:, 0],
                        positions=reactantnxyz[:, 1:],
                        cell=reactantlattice,
                        pbc=True)
            product = Atoms(productnxyz[:, 0],
                        positions=productnxyz[:, 1:],
                        cell=productlattice,
                        pbc=True)
            
        else:
            reactant = Atoms(reactantnxyz[:, 0],
                        positions=reactantnxyz[:, 1:])
            product = Atoms(productnxyz[:, 0],
                        positions=productnxyz[:, 1:])

        num_images = params["neb_params"]["num_images"]
        images = [reactant]
        for i in range(int(num_images-2)):
            images.append(reactant.copy())
        images.append(product)

        neb = NEB(images)
        # # idpp interpolation
        # neb.interpolate('idpp', mic=True)
        # linear interpolation
        neb.interpolate(mic=True)

        images = []
        for idx, image in enumerate(neb.images):
            images.append(image)

    else:
        coord_set = params['image_coords']
        if 'image_lattices' in params:
            lattice_set = params['image_lattices']
            images = []

            for coords, lattice in zip(coord_set, lattice_set):
                nxyz = coords_to_xyz(coords=coords).astype('float')
                atoms = Atoms(nxyz[:, 0],
                            positions=nxyz[:, 1:],
                            cell=lattice,
                            pbc=True)
                images.append(atoms)
        else:
            images = []

            for coords in coord_set:
                nxyz = coords_to_xyz(coords=coords).astype('float')
                atoms = Atoms(nxyz[:, 0],
                            positions=nxyz[:, 1:])
                images.append(atoms)

    return images


def get_model_path(params):
    if all(['weightpath' in params, 'nnid' in params]):
        full_path = os.path.join(params['weightpath'],
                                 str(params['nnid']))
        if not os.path.isdir(full_path):
            full_path = os.path.join(params['mounted_weightpath'],
                                     str(params['nnid']))
    else:
        full_path = params['model_path']

    return full_path


def get_calc_kwargs(params):
    model_path = get_model_path(params)
    calc_kwargs = {**params["calc_kwargs"],
                   "model_path": model_path}
    return calc_kwargs


def init_images(params):

    calc_kwargs = get_calc_kwargs(params)
    images = get_images(params)

    neb_params = params["neb_params"]
    atoms_kwargs = params["atoms_kwargs"]

    opt_start_end(images=images,
                  neb_params=neb_params,
                  calc_kwargs=calc_kwargs,
                  atoms_kwargs=atoms_kwargs)

    neb = get_neb(images=images,
                  neb_params=neb_params,
                  atoms_kwargs=atoms_kwargs,
                  calc_kwargs=calc_kwargs)

    return neb, images


def save_bands(neb):
    """
    Save the first and last bands with the proper energies and forces
    """

    num_images = len(neb.images)
    num_atoms = neb.num_atoms[:1]
    calculator = neb.calculator
    atoms_kwargs = neb.atoms_kwargs

    trj = Trajectory(NEB_TRJ)
    first_band = trj[:num_images]
    last_band = trj[-num_images:]

    loaded_bands = [first_band, last_band]

    for i, loaded_band in enumerate(loaded_bands):
        # Do it without batching so that everything will have the proper
        # splitting of forces

        trj_path = FINAL_TRJ.format(num=i)
        trj = Trajectory(trj_path, mode='w')

        for image in loaded_band:
            atoms_batch = make_atoms_batch(calculator=calculator,
                                           atoms_kwargs=atoms_kwargs,
                                           images=[image],
                                           num_atoms=num_atoms)
            get_ens_forces(atoms_batch=atoms_batch,
                           num_atoms=num_atoms)
            trj.write(atoms_batch)


def add_convg_info(converged):

    with open(NEB_LOG, 'a') as f:
        f.write(TERMINATION_LINE)
        if converged:
            f.write(CONVERGED_LINE)
        else:
            f.write(NON_CONVERGED_LINE)


def run_neb(params):
    """
    Consider making the neighbor list update more efficient or less often
    """

    neb, images = init_images(params)

    neb_params = params["neb_params"]
    optimizer = get_optim(atoms=neb,
                          optim_name=neb_params["optim_name"],
                          optim_kwargs=neb_params["optim_init_kwargs"],
                          logfile=NEB_LOG,
                          trj=NEB_TRJ)

    optimizer.run(**neb_params['optim_run_kwargs'])

    save_bands(neb)
    add_convg_info(converged=optimizer.converged())


def load_defaults():
    direc = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(direc, 'default_details.json')

    with open(default_path, 'r') as f:
        info = json.load(f)

    return info


def load_params(file):
    info = load_defaults()
    with open(file, 'r') as f:
        new_info = json.load(f)
    info.update(new_info)

    if "details" in info:
        info.update(info["details"])
        info.pop("details")

    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
                        help=('The path with to the config file'),
                        default='job_info.json')
    args = parser.parse_args()

    params = load_params(file=args.config_file)
    run_neb(params)


if __name__ == "__main__":
    main()

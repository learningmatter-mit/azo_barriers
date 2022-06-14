# Thermal barriers of azobenzene derivatives

This repository contains code for computing the thermal barriers of azobenzene derivatives. The code base uses a neural network force field (NFF) to compute energies and gradients. These are in turn used to optimize reactants, products, transition states, and singlet-triplet crossing points for intersystem crossing.

This code repository is developed in the Learning Matter Lab (led by prof. Rafael Gomez-Bombarelli) at MIT.

## Conda environment

We recommend creating a [conda](https://conda.io/docs/index.html) environment to run the code. You can learn more about managing anaconda environments by reading [this page](http://conda.pydata.org/). To create the environment, use the following commands:

```bash
conda upgrade conda
conda env create -f environment.yml
```

To ensure that the `barriers` environment is accessible through Jupyter, add the the `barriers` display name:
```bash
python -m ipykernel install --user --name barriers --display-name "Python [conda env:barriers"]
```

Next, download the [Neural Force Field](https://github.com/learningmatter-mit/NeuralForceField) repository, which is also managed by our group. You can either install it through `pip`, or clone it and put the folder in your path (see below). We recommend the latter, since NFF is under constant development, and often needs to be pulled from github when changes are made.

Lastly, put the `azo_barriers` repository (and possibly NFF) in your path by adding the following lines to `~/.bashrc` (linux) or `~/.bash_profile` (mac):

```
# add the `azo_barriers` path
export BARRIERS=<path to azo_barriers>
export PYTHONPATH=$BARRIERS:$PYTHONPATH

# if you're adding the Neural Force Field path as well
export NFFDIR=<path to NFF>
export PYTHONPATH=$NFFDIR:$PYTHONPATH
```
## Tutorials
[Jupyter notebook tutorials](https://github.com/learningmatter-mit/azo_barriers/tree/main/tutorials) show how to load and interpret our published barrier data, which can be found [here](https://doi.org/10.18126/unc8-336t). They also show how to do the same for any data you may generate yourself. To learn how to generate your own data, see the **Examples** section in this document.

## Examples

An example calculation can be found in the `examples` folder. To test it out, run the following code on the command line:
```
cd examples
./run.sh
```
This should produce a series of calculations for two different molecules. The calculations include:
- Initial 3D structure generation through RDKit
- 4 relaxed scans per molecule to generate 4 possible transition states (TSs) of different mechanisms
- Metadynamics-based conformer generation to generate reactant, product, and TS conformers 
- Eigenvector following to optimize the TSs
- Hessian calculations on the optimized reactants and products
- Singlet-triplet minimum energy crossing point search
- Intrinsic reaction coordinate generation for the optimized TSs

To run this for your own molecules, simply supply their SMILES strings in the file `examples/job_info.json`:
```
"smiles_list": [...]
````
Note that you only need to provide one cis or trans SMILES per molecule. You can also set the directory of your singlet neural network model (`weightpath`), the directory of your triplet model `(triplet_weightpath`), the device you want to use (`cpu` if you have no GPUs, or the index of the GPU you want to use), and the number of parallel jobs to run at once for each of the configs (`num_parallel`).

The final results are stored in `examples/summary.pickle`. [Tutorials](https://github.com/learningmatter-mit/azo_barriers/tree/main/tutorials) show how to load, visualize, and interpret the results. They also go into some detail about other parameters you can specify in `job_info.json`

## Pre-trained models
A set of pre-trained models can be found in `models`.

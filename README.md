# Thermal barriers of azobenzene derivatives

This repository contains code for computing the thermal barriers of azobenzene derivatives. The code base uses a neural network force field (NFF) to compute energies and gradients. These are in turn used to optimize reactants, products, transition states, and singlet-triplet crossing points for intersystem crossing.

This code repository is developed in the Learning Matter Lab (led by prof. Rafael Gomez-Bombarelli) at MIT.

## Conda environment

We recommend creating a [conda](https://conda.io/docs/index.html) environment to run the code. You can learn more about managing conda environments by reading [this page](http://conda.pydata.org/). 

Run `./setup.sh` to create the environment. Then run `source ~/.bashrc` (linux) or `source ~/.bash_profile` (Mac). You're now ready to use the code base! If something goes wrong, please see [this file](https://github.com/learningmatter-mit/azo_barriers/blob/main/setup.md).


## Tutorials
[Jupyter notebook tutorials](https://github.com/learningmatter-mit/azo_barriers/tree/main/tutorials) show how to load and interpret our published barrier data. The data can be found [here](https://doi.org/10.18126/unc8-336t). The tutorials also show how to load any data you may generate yourself. To learn how to generate your own data with pre-made scripts, see the **Examples** section in this document.

## Examples

An example calculation can be found in the `examples` folder. To test it out, run the following code on the command line:
```
cd examples
./run.sh
```

To run this for your own molecules, simply supply their SMILES strings in the file `examples/job_info.json`:
```
"smiles_list": [...]
````

The script should produce a series of calculations for two different molecules. The calculations include:
- Initial 3D structure generation through RDKit
- 4 relaxed scans per molecule to generate 4 possible transition states (TSs) of different mechanisms
- Metadynamics-based conformer generation to generate reactant, product, and TS conformers 
- Eigenvector following to optimize the TSs
- Hessian calculations on the optimized reactants and products
- Singlet-triplet minimum energy crossing point search
- Intrinsic reaction coordinate generation for the optimized TSs
- Single point $S_0/S_1$ gap calculations on the optimized *cis* and *trans* geometries

Note that you only need to provide one cis or trans SMILES per molecule. Optionally, can also change the directory of your singlet neural network model (`weightpath`), the directory of your triplet model (`triplet_weightpath`), the directory of your $S_0/S_1$ gap model (`s0_s1_weightpath`), the device you want to use (`cpu` if you have no GPUs, or the index of the GPU you want to use), and the number of parallel jobs to run at once for each of the configs (`num_parallel`).

The final results are stored in `examples/summary.pickle`. [Tutorials](https://github.com/learningmatter-mit/azo_barriers/tree/main/tutorials) show how to load, visualize, and interpret the results. They also go into some detail about other parameters you can specify in `job_info.json`

## Pre-trained models
A set of pre-trained models can be found in `models`.

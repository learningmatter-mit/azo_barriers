# Thermal barriers of azobenzene derivatives

This repository contains code for computing the thermal barriers of azobenzene derivatives. The code base uses a neural network force field (NFF) to compute energies and gradients. These are in turn used to optimize reactants, products transition states, and singlet-triplet crossing points for intersystem crossing.

This code repository is developed in the Learning Matter Lab (led by prof. Rafael Gomez-Bombarelli) at MIT.

## Conda environment

We recommend creating a conda(https://conda.io/docs/index.html) environment to run the code. You can learn more about managing anaconda environments by reading [this page](http://conda.pydata.org/). To create the environment, use the following commands:

```bash
conda upgrade conda
conda env create -f environment.yml
```

Next, download the [Neural Force Field](https://github.com/learningmatter-mit/NeuralForceField) repository, which is also managed by our group. You can either install it through `pip`, or clone it and put the folder in your path (see below). We recommend the latter, since NFF is under constant development, and often needs to be pulled from github when changes are made.

Lastly, put the `azo_barriers` repository in your path by adding the following lines to ~/.bashrc (linux) or ~/.bash_profile (mac):

```
# add the `azo_barriers` path
export BARRIERS=<path to azo_barriers>
export PYTHONPATH=$BARRIERS:$PYTHONPATH

# if you're adding the Neural Force Field path as well
export NFFDIR=<path to NFF>
export PYTHONPATH=$NFFDIR:$PYTHONPATH
```


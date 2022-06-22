## Setup

If the `setup.sh` script does not work, you can perform the following steps, which is what `setup.sh1` is doing under the hood. 


To create the environment, use the following commands:

```bash
conda upgrade conda
conda env create -f environment.yml
```

This creates an environment called `barriers`. 

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

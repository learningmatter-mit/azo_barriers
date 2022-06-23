#!/bin/bash -i

source deactivate > /dev/null 2>&1
exists='True'
source activate barriers || exists='False' > /dev/null 2>&1
source deactivate > /dev/null 2>&1

if [ "$exists" == "True" ]; then
    echo 'Barriers conda environment already exists; skipping environment installation'
else
    # upgrade conda
    conda upgrade conda
    # install environment
    conda env create -f environment.yml
fi

# display environment in jupyter
python -m ipykernel install --user --name barriers --display-name "Python [conda env:barriers"]

barrier_dir=$(pwd)

# install NFF

cd ..
if [ -d "NeuralForceField" ]; then
    echo 'NeuralForceField directory already exists; skipping installation'
else
    git clone git@github.com:learningmatter-mit/NeuralForceField.git
fi
nff_dir=$(pwd)/NeuralForceField

cd -


# export paths to bashrc or bash_profile
text='export BARRIERS='$barrier_dir'
export PYTHONPATH=$BARRIERS:$PYTHONPATH
export NFFDIR='$nff_dir'
export PYTHONPATH=$NFFDIR:$PYTHONPATH'

if [ -f -a ~/.bashrc ]; then
    echo "Found ~/.bashrc. Exporting path."
    bash_path="$HOME/.bashrc"
elif [ -f -a ~/.bash_profile ]; then
    echo "Found ~/.bash_profile. Exporting path."
    bash_path="$HOME/.bash_profile"
fi

if [ ! -z "${bash_path}" ]; then
    while IFS= read -r line; do
        grep "$line" $bash_path || echo "$line" >> $bash_path
    done <<< "$text"
    echo "Done!"
else
    echo "Couldn't find ~/.bashrc or ~/.bash_profile. Please export paths in the appropriate setup file."
fi

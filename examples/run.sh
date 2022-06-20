#!/bin/bash
source activate barriers

# get the directory of this script
direc="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

python $direc/../barriers/utils/workflow.py

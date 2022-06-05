source activate barriers

# get the directory of this script
direc="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

# run the script
python $direc/../../barriers/triplet_crossing/neural_triplet_crossing.py  --info_file job_info.json > neural_triplet_crossing.log


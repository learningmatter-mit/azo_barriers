source activate barriers

# get the directory of this script
direc="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

# run the script
python $direc/../../barriers/relaxed_scan/neural_relaxed_scan.py  --info_file job_info.json > neural_relaxed_scan.log


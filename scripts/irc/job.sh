source activate barriers

# get the directory of this script
direc="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

# run the script
python $direc/../../barriers/irc/neural_irc.py  --info_file job_info.json > neural_irc.log


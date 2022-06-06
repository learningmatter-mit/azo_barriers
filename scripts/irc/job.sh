source activate barriers

# get the directory of this script
direc="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

# render the info file using `default_details.json`, updated with whatever is in `job_info.json`
python $direc/../../barriers/utils/render_info.py --cwd $(pwd) --script_dir $direc

# run the script
python $direc/../../barriers/irc/neural_irc.py # > neural_irc.log

# remove the ASE vib folder
rm -r vib

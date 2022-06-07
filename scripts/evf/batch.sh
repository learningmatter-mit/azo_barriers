# get the directory of this script
direc="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

# create sub-directories and a `tasks.txt` file with information about the jobs to run in each sub-directory
python $direc/../../barriers/utils/batch.py --cwd $(pwd) --script_dir $direc/job.sh

# change to the number of jobs you want to run in parallel
parallel -j 2 < tasks.txt


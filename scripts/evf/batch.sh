source activate barriers

# get the directory of this script
direc="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

# render the info files using `default_details.json`, updated with whatever is in `job_info.json`
folders=$(ls -d */)
for folder in ${folders[@]}; do
        python $direc/../../barriers/utils/render_info.py --cwd $folder --script_dir $direc
done

# create sub-directories and a `tasks.txt` file with information about the jobs to run in each sub-directory
python $direc/../../barriers/utils/batch.py --cwd $(pwd) --script_dir $direc/job.sh

# get number of parallel processes requested
folder=$(ls -d */ | head -n 1)
num_parallel=$(cat $folder/job_info.json | jq ".num_parallel")
# run in parallel with GNU parallel
parallel -j $num_parallel < tasks.txt


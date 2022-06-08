source activate barriers

# get the directory of this script
direc="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

# render the info files using `default_details.json`, updated with whatever is in `job_info.json`
folders=$(ls -d */)
for folder in ${folders[@]}; do
	python $direc/../../barriers/utils/render_info.py --cwd $folder --script_dir $direc
done


folder=$(ls -d */ | head -n 1)
# get number of chunks in which to split the jobs
num_chunk=$(cat $folder/job_info.json | jq ".num_in_chunk")
# get number of parallel jobs to be run at once
num_parallel=$(cat $folder/job_info.json | jq ".num_parallel")

# split folders into chunks to do together
text=$(python $direc/split_folders.py --job_dir "." --batch_size $num_chunk )
cwd=$(pwd)


while IFS= read -r line; do

	scratch="/tmp/$(date +%Y%m%d%H%M%S%N)"

	# Create scratch folder

	mkdir -p  $scratch
	echo "Copying chunk of jobs to $scratch"

	# move the directories in this chunk to scratch
	sub_dirs=($line)
	for folder in ${sub_dirs[@]}; do
		mv $folder $scratch/
	done

	# go to scratch
	cd $scratch

	clean_up()
	{
		echo 'Bringing output back'
		shopt -s extglob
		cd $cwd

		mv $scratch/*  .
		rm -r $scratch

		shopt -u extglob
		echo 'Finished cleaning up'
	}

	python $direc/../../barriers/confgen/batched_neural_confgen.py --info_file job_info.json --np $num_parallel > neural_confgen.log
	clean_up

done <<< "$text"


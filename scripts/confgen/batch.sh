source activate barriers

# get the directory of this script
direc="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

# render the info files using `default_details.json`, updated with whatever is in `job_info.json`
folders=$(ls -d */)
for folder in ${folders[@]}; do
	python $direc/../../barriers/utils/render_info.py --cwd $folder --script_dir $direc
done


# do a certain number of chunks at a time so that if the job doesn't finish,
# we still get results from lots of molecules

# change --batch_size to the number of chunks you want to do at a time

text=$(python $direc/split_folders.py --job_dir "." --batch_size 100 )
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
		rm -r $scfolder

		shopt -u extglob
		echo 'Finished cleaning up'
	}

	# change --np to the number of jobs you want to run in parallel at once
	python $direc/../../barriers/confgen/batched_neural_confgen.py --info_file job_info.json --np 50 > neural_confgen.log
	clean_up

done <<< "$text"


source activate barriers

# get the directory of this script
direc="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
parallel=$direc/../../ext_programs/parallel

# change to the number of jobs you want to run in parallel
$parallel -j 4 < tasks.txt 


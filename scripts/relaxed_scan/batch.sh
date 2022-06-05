{% set platform = jobspec.details.compute_platform | default("engaging") -%}
{% set use_gpu = (jobspec.details.device | default(0)) != "cpu" -%}
{% set partitions = jobspec.details.partitions -%}
#!/bin/bash
#SBATCH -n {{jobspec.details.nprocs | default(8)}}
#SBATCH -N 1
#SBATCH -t {{jobspec.details.max_time | default(9600)}}
#SBATCH -p {{jobspec.details.partitions|join(",")}}
{% if jobspec.details.req_mem | default(true) -%}
#SBATCH --mem-per-cpu={{jobspec.details.maxcore | default(1000)}}
{% endif -%}
#SBATCH --no-requeue
#SBATCH --signal=B:2@300
{% if platform == "supercloud" and use_gpu -%}
#SBATCH --gres=gpu:volta:1
{% elif use_gpu -%}
#SBATCH --gres=gpu:{{jobspec.details.gres}}
{% endif -%}
{% if platform in ['engaging', none] -%}
#SBATCH --constraint=centos7
{% endif -%}
{% if jobspec.details.exclusive -%}#SBATCH --exclusive{% endif %}
{% if jobspec.details.get('compute_platform') == "engaging" -%}
#SBATCH --exclude=node[1034-1035]
{% endif -%}

source ~/.bashrc

{% if jobspec.details.get('compute_platform') == "supercloud" %}
. /etc/profile.d/modules.sh
module load anaconda/2021b
{% elif jobspec.details.get("conda_module_load") -%}
module load {{jobspec.details.conda_module_load}}
{% endif %}

export parallel={{jobspec.details.gnu_parallel | default("/home/gridsan/saxelrod/apps/parallel-20210822/bin/parallel") }}
$parallel -j {{jobspec.num_parallel}} < tasks.txt & pid=$!

clean_up()
{
kill -9 $pid
echo "Interrupted"
}

trap 'clean_up;  exit 1' SIGTERM SIGINT
wait $pid




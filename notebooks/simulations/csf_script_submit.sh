#!/bin/bash
#$ -j Y
#$ -cwd
#$ -V

# Get the directory of this script.
output_name="output_CS_nvx1000_noisept5_steps500"
config_name="config.yml"
this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
nr_jobs=1
output_file=$this_dir/$output_name
if [ ! -d "$output_file" ]; then
    mkdir "$output_file"
fi
job="qsub -b y -j y -q cuda.q@jupiter -pe smp ${nr_jobs} -wd ${output_file} -N ${output_name} -o ${output_name}.txt"
# Path to the python script to be executed by the job
script_path="${this_dir}/CSENF_script_test.py"
cp $this_dir/$config_name $output_file/$config_name
# Submit the job, explicitly calling the current conda python interpreter.
${job} "$(which python)" "${script_path} --output ${output_name} --config ${config_name}"

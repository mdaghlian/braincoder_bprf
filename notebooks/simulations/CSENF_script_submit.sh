#!/bin/bash
#$ -j Y
#$ -cwd
#$ -V

# Get the directory of this script.
this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
nr_jobs=1
job_name="ctest"

# Build the qsub command.
# Note the addition of the -b y flag to indicate a binary job.
job="qsub -b y -q cuda.q@jupiter -pe smp ${nr_jobs} -wd ${this_dir} -N ${job_name} -o ${job_name}.txt"

# Path to the python script to be executed by the job
script_path="${this_dir}/CSENF_script_test.py"
echo $script_path
exit 1
# Submit the job, explicitly calling the current conda python interpreter.
${job} "$(which python)" "${script_path}"

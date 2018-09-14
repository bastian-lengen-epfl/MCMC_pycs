#!/bin/bash -l

# ***** Script that launches slurm jobs *****

# set the simulation basename and the file you want to run


export JOB_BASENAME="3a"
export SERVER_NAME="regor"  # on Regor : 'r3', 'r4' ; otherwise 'deneb' or 'fidis'
export QUEUE_NAME="r4"  # on Regor : 'r3', 'r4' ; on deneb "debug" or "serial"

export OBJECT_NAME='HE0435b'
export NUM_CORE="16"
max_node=1


start_file="start_3a.slurm"
job_name_csr=$OBJECT_NAME$JOB_BASENAME

echo "job name = "$job_name_csr
echo "start file = "$start_file
echo "max node = "$max_node
echo "MCMC_PyCS : sending job as sbatch -J $job_name_csr -n $NUM_CORE (-N $max_node) (-p $QUEUE_NAME) $start_file"

sbatch -J $job_name_csr -p $QUEUE_NAME -n $NUM_CORE -N 1 $start_file

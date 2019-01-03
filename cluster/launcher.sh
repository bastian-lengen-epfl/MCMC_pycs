#!/bin/bash -l

# ***** Script that launches slurm jobs *****

# set the simulation basename and the file you want to run

export JOB_BASENAME="3all"
export SERVER_NAME="lesta"  # on Regor : 'r3', 'r4' ; otherwise 'deneb' or 'fidis'
export QUEUE_NAME="p4"  # on Regor : 'r3', 'r4' ; on deneb "debug" or "serial"

export OBJECT_NAME='J0246'
export DATA_NAME='ECAM'
export NUM_CORE="12"
export WORK_DIR='./'


max_node=1


start_file="start_3all.slurm"
job_name_csr=$OBJECT_NAME$JOB_BASENAME

echo "job name = "$job_name_csr
echo "start file = "$start_file
echo "max node = "$max_node
echo "MCMC_PyCS : sending job as sbatch -J $job_name_csr -n 1 -c $NUM_CORE (-p $QUEUE_NAME) $start_file"

sbatch -J $job_name_csr -p $QUEUE_NAME -n $NUM_CORE -N 1 $start_file

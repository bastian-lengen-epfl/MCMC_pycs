#!/bin/bash -l
#SBATCH -o slurm_%j.out  # STDOUT
#SBATCH -e slurm_%j.err  # STDERR
#SBATCH --exclude=node[222,224]

date
set -x

cd ..

run_command="srun -n 1 -c $NUM_CORE -u"
run_options="--mpi=pmi2"
exec_command="python 3a_generate_tweakml.py"
exec_options=""
whole_command="$run_command $run_options $exec_command $exec_options"

echo "running '$whole_command'"
eval $whole_command

date

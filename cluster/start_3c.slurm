#!/bin/bash -l

#SBATCH -o slurm_%j.out  # STDOUT
#SBATCH -e slurm_%j.err  # STDERR

date
set -x
cd ..

exec_command="python 3c_optimise_copy_mocks.py"
exec_options=""
whole_command="$exec_command $exec_options"

echo "running '$whole_command'"
eval $whole_command

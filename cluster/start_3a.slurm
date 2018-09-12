#!/usr/bin/env bash
####SBATCH --time=5-00:00:00  #days-hours:minutes:seconds or hours:minutes:seconds
#SBATCH -o slurm_%j.out  # STDOUT
#SBATCH -e slurm_%j.err  # STDERR
####SBATCH --exclude=node[222,224,230,231,232]

cd ..

exec_command="python 3a_generate_tweakml.py"
exec_options=""
whole_command="$exec_command $exec_options"

echo "running '$whole_command'"
eval $whole_command
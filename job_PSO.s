#!/bin/bash
#SBATCH -p r4 -c 16 -o ./simPSO.stdout -e ./simPSO.stderr
# run the simulation
python find_param_PSO.py
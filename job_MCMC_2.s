#!/bin/bash
#SBATCH -p r4 -c 16 -o ./sim2.stdout -e ./sim2.stderr
# run the simulation
python find_param_metro.py

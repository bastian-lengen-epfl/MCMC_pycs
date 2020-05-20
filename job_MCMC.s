#!/bin/bash
#SBATCH -p r4 -c 16 -o ./sim.stdout -e ./sim.stderr
# run the simulation
python find_param_metro.py

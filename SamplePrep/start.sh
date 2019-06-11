#! /bin/bash

# Script to run everything
# Source environment/bashrc
source ~/.bashrc

# Replace the following with your own environment location
source activate /home/cbilodeau/anaconda2/envs/deepchem2

module load gromacs/453


# 1 Clean Files:
sh clean.sh
sh SASA_Calc.sh
python SASA_Calc.py


# 2 Create Occupancy Maps:
python VoxelParse.py

# 3 Divide Into Patches:
python PartitionPatches.py

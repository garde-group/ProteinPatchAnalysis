#!/bin/sh
#SBATCH -t 355
#SBATCH --partition dcs
#SBATCH --nodes 1
#SBATCH --gres=gpu:4

path_binary=/gpfs/u/home/MMBS/MMBScmbl/scratch/tensorflow/ProteinPatchAnalysis/ModelOptimization

module load xl_r mpich cuda

srun --gres=gpu:4 -n 4 ./bindProcessToGpu.sh python $path_binary/OptimizeParams.py 

set +x




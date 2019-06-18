#!/bin/sh
#SBATCH -t 355
#SBATCH --partition dcs
#SBATCH --nodes 1
#SBATCH --gres=gpu:4

path_binary=/gpfs/u/home/MMBS/MMBScmbl/scratch/tensorflow/ProteinPatchAnalysis/DCS_GPU_PracticeRun

module load xl_r mpich cuda


for batch in 256 #128 256 32 64 #128 256
do

rm -rf batch-var-run.py
sed "s/xxx/$batch/g" batch-var.py > batch-var-run.py

srun --gres=gpu:4 -n 4 ./bindProcessToGpu.sh python $path_binary/batch-var-run.py 

set +x
done




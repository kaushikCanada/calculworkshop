#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1 # request a GPU
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-01:00:00
#SBATCH --output=log.out
#SBATCH --account=def-sh1352
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=kaushik.roy@inrs.ca

module load python/3.11 # Using Default Python version - Make sure to choose a version that suits your application

echo "Hello World"
nvidia-smi

source ~/royenv/bin/activate

export TORCH_NCCL_BLOCKING_WAIT=1

echo "starting training..."
srun python ~/scratch/calculworkshop/cifar10.py --batch_size=256 --num_workers=2

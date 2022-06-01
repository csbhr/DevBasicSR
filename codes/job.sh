#!/bin/bash
#SBATCH -J Dev
#SBATCH -p defq
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH -t 9999
#SBATCH -o slurm-Develop.out
#SBATCH --gres=gpu:2

module load cuda11.1/

source activate pytorch1.11.0
python setup.py develop

python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt options/train/Develop/train_Develop.yml --launcher pytorch

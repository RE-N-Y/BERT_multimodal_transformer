#!/bin/sh

#SBATCH -p gpu --gres=gpu -C K80
#SBATCH --mem=20gb
#SBATCH -c 4
#SBATCH -a 0-2
#SBATCH -t 0-71:00:00  
#SBATCH -J slee232
#SBATCH -o /scratch/slee232/output/output-%j
#SBATCH -e /scratch/slee232/output/error-%j

module load anaconda3/5.3.0b
source activate /home/slee232/.conda/envs/pydeep

wandb agent nyre/MAG_HUMOR/6mnek6ug
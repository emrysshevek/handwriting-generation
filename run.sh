#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mem 16000M
#SBATCH --time 72:00:00
#SBATCH -C 'rhel7'
#SBATCH --mail-user=masonfp@byu.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#%Module

module purge
module load cuda

export PATH="/fslhome/masonfp/fsl_groups/fslg_hwr/compute/env/generation/bin:$PATH"
which python

cd "/fslhome/masonfp/fsl_groups/fslg_hwr/compute/handwriting-generation"
python -u train.py

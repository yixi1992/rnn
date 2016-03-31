#!/bin/bash

#SBATCH -t 10:00:00
#SBATCH -N 1

source /lustre/yixi/decouplednet/DecoupledNet/inference/load_deeplab_dependencies.sh
source ./load_deeplab_dependencies.sh

python ./solve.py > solve.out

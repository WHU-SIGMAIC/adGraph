#!/bin/bash
#SBATCH -J adGraph
#SBATCH -p kshdexcluxd
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -o %j-%x.log
#SBATCH -e %j-%x.log
#SBATCH --gres=dcu:1

echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"

cd ${SLURM_SUBMIT_DIR}

module purge
module load compiler/gcc/8.2.0
module load compiler/dtk/24.04
module load compiler/cmake/3.23.3
module list

source ./setup_hip_env.sh

./build_hip.sh -v

date

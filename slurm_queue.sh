#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --nodes=1                # Number of nodes - keep nodes to minimum to avoid overheads for data transfer between nodes
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --mem=32G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of allocated gpus per node
#SBATCH --cpus-per-task=8        # Number of CPUs per process
# Copied from slurm_queue.sh from the s4 repo

echo $partition
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_ADDR="$MASTER_ADDR

opts=( $OPTS ) # Get all python arguments

if [ -z $PYFUNC ]; then
  PYFUNC=train.train
fi

# Do normal work here:
cd $HOME/ssm/mamba
module load cuda/12.8.0
module load gcc/11.5.0

# Activate conda env:
ENV_NAME="mamba"
source activate $ENV_NAME

# Log input arguments
echo "Input arguments: "${opts[@]}

# SLURM settings:
walltime=$(squeue -h -j $SLURM_JOBID -o "%l")
partition=$(squeue -h -j $SLURM_JOBID -o %P)

# Echos
echo '==================================='
echo 'SLURM SETTINGS:'
echo '---------------'
echo 'Job ID:                     '$SLURM_JOBID
echo 'Walltime:                   '$walltime
echo 'Partition:                  '$partition
echo 'Number of nodes:            '$SLURM_NNODES
echo 'Number of tasks per node:   '$SLURM_NTASKS_PER_NODE
echo 'Number of GPUs per node:    '$SLURM_GPUS_ON_NODE
echo 'CPUs per task:              '$SLURM_CPUS_PER_TASK
echo 'Python function:            '$PYFUNC
echo '==================================='

# Call training script with optional arguments (defaults specified in train script):
# python $PYFUNC "${opts[@]}"
srun python -m $PYFUNC "${opts[@]}"

#!/bin/bash
#SBATCH --job-name=TAOKD # 作业名称
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4090:1

JOB_ID=$SLURM_JOB_ID
NODE_NAME=$(hostname)

mkdir -p log

OUTPUT_FILE="log/TAOKD-${JOB_ID}-${NODE_NAME}.log"

echo "Job started at: $(date)" >> $OUTPUT_FILE
echo "Running on node: $NODE_NAME" >> $OUTPUT_FILE
echo "Using GPUs:" >> $OUTPUT_FILE
nvidia-smi >> $OUTPUT_FILE

source activate kd

srun python pretrainer.py >> $OUTPUT_FILE 2>&1

echo "Job ended at: $(date)" >> $OUTPUT_FILE

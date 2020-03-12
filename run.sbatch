#!/bin/sh

#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --time=0:30:00
#SBATCH --job-name="nlu_hw4"

module purge
module load anaconda3/5.3.1
module load cuda/10.0.130
module load gcc/6.3.0

# Replace with your NetID
NETID=jh6926
source activate /scratch/${NETID}/nlu/env

DATA_DIR=/scratch/${NETID}/nlu/BoolQ
TASK_NAME=boolq

cd /scratch/${NETID}/nlu/code/transformers
export PYTHONPATH=/scratch/${NETID}/nlu/code/transformers/src:$PYTHONPATH
python ./examples/run_glue.py \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --task_name ${TASK_NAME} \
    --do_train \
    --do_eval \
    --data_dir ${DATA_DIR} \
    --max_seq_length=128 \
    --per_gpu_eval_batch_size=8 \
    --per_gpu_train_batch_size=8 \
    --learning_rate 2e-5 \
    --num_train_epoch 3.0 \
    --output_dir /scratch/${NETID}/nlu/${TASK_NAME}_run/
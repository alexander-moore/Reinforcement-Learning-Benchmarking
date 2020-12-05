#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -p short
#SBATCH -C V100|P100|K80
#SBATCH -o train.out
#SBATCH --gres=gpu:1
python main.py --train_dqn --agent double_dqn.SampleAgent --model sample_model.SampleModel --run_name training_run --env_name Breakout-v0

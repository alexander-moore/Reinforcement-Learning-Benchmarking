#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -C K20
#SBATCH -o train.out

module load anaconda/anaconda3
source activate Proj3

python main.py --train_dqn --agent sample_agent.SampleAgent --model sample_model.SampleModel --run_name training_run --num_episodes 30000 --lr 0.0001
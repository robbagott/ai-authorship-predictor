#!/bin/sh

#SBATCH -p gpu --gres=gpu:1
#SBATCH --constraint=geforce3090

#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH -t 12:00:00

#SBATCH -J ai-authorship-predictor-knns

#SBATCH -o train-knn-%j.out
#SBATCH -e train-knn-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexander_meyerowitz@brown.edu

module load python/3.9.0
module load gcc/10.2 cuda/11.7.1 cudnn/8.2.0
source /users/ameyerow/ai-authorship.venv/bin/activate

cd /users/ameyerow/ai-authorship-predictor-training-script

for f in *.pt; do
  python train_knn.py --model-file ${f%.*}.pt --output-file knns/${f%.*}-knn.pt --results-file results/${f%.*}-results.txt
done
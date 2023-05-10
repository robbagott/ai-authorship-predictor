#!/bin/sh

#SBATCH -p gpu --gres=gpu:1
#SBATCH --constraint=geforce3090

#SBATCH -n 4
#SBATCH --mem=10G
#SBATCH -t 12:00:00

#SBATCH -J ai-authorship-predictor

#SBATCH -o train-%j.out
#SBATCH -e train-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexander_meyerowitz@brown.edu

module load python/3.9.0
module load gcc/10.2 cuda/11.7.1 cudnn/8.2.0
source /users/ameyerow/ai-authorship.venv/bin/activate

cd /users/ameyerow/ai-authorship-predictor-training-script

# all
python train.py --output-file contrast_nf_1234_4e.pt --data-option 1234 --no-freeze --loss contrast --batch-size 16 --epochs 4
# No easy negatives
python train.py --output-file contrast_nf_134_4e.pt --data-option 134 --no-freeze --loss contrast --batch-size 16 --epochs 4
# No triplets with both random pos+neg
python train.py --output-file contrast_nf_123_4e.pt --data-option 123 --no-freeze --loss contrast --batch-size 16 --epochs 4
# No hard negatives
python train.py --output-file contrast_nf_14_4e.pt --data-option 14 --no-freeze --loss contrast --batch-size 16 --epochs 4
# Only semi-hard negative and hard negatives
python train.py --output-file contrast_nf_13_4e.pt --data-option 13 --no-freeze --loss contrast --batch-size 16 --epochs 4

# all
python train.py --output-file triplet_nf_1234_4e_10a.pt --data-option 1234 --no-freeze --loss triplet --batch-size 16 --epochs 4 --alpha 10
# No easy negatives
python train.py --output-file triplet_nf_134_4e_10a.pt --data-option 134 --no-freeze --loss triplet --batch-size 16 --epochs 4 --alpha 10
# No triplets with both random pos+neg
python train.py --output-file triplet_nf_123_4e_10a.pt --data-option 123 --no-freeze --loss triplet --batch-size 16 --epochs 4 --alpha 10
# No hard negatives
python train.py --output-file triplet_nf_14_4e_10a.pt --data-option 14 --no-freeze --loss triplet --batch-size 16 --epochs 4 --alpha 10
# Only semi-hard negative and hard negatives
python train.py --output-file triplet_nf_13_4e_10a.pt --data-option 13 --no-freeze --loss triplet --batch-size 16 --epochs 4 --alpha 10

# all
python train.py --output-file mixed_nf_1234_4e.pt --data-option 1234 --no-freeze --loss mixed --batch-size 16 --epochs 4
# No easy negatives
python train.py --output-file mixed_nf_134_4e.pt --data-option 134 --no-freeze --loss mixed --batch-size 16 --epochs 4
# No triplets with both random pos+neg
python train.py --output-file mixed_nf_123_4e.pt --data-option 123 --no-freeze --loss mixed --batch-size 16 --epochs 4
# No hard negatives
python train.py --output-file mixed_nf_14_4e.pt --data-option 14 --no-freeze --loss mixed --batch-size 16 --epochs 4
# Only semi-hard negative and hard negatives
python train.py --output-file mixed_nf_13_4e.pt --data-option 13 --no-freeze --loss mixed --batch-size 16 --epochs 4

#!/bin/sh

# all
python train.py --output-file contrast_nf_1234_1e.pt --data-option 1234 --no-freeze --loss contrast --batch-size 16 --epochs 1
# No easy negatives
python train.py --output-file contrast_nf_134_1e.pt --data-option 134 --no-freeze --loss contrast --batch-size 16 --epochs 1
# No triplets with both random pos+neg
python train.py --output-file contrast_nf_123_1e.pt --data-option 123 --no-freeze --loss contrast --batch-size 16 --epochs 1
# No hard negatives
python train.py --output-file contrast_nf_14_1e.pt --data-option 14 --no-freeze --loss contrast --batch-size 16 --epochs 1
# Only semi-hard negative and hard negatives
python train.py --output-file contrast_nf_13_1e.pt --data-option 13 --no-freeze --loss contrast --batch-size 16 --epochs 1

# all
python train.py --output-file contrast_nf_1234_8e.pt --data-option 1234 --no-freeze --loss contrast --batch-size 16 --epochs 8
# No easy negatives
python train.py --output-file contrast_nf_134_8e.pt --data-option 134 --no-freeze --loss contrast --batch-size 16 --epochs 8
# No triplets with both random pos+neg
python train.py --output-file contrast_nf_123_8e.pt --data-option 123 --no-freeze --loss contrast --batch-size 16 --epochs 8
# No hard negatives
python train.py --output-file contrast_nf_14_8e.pt --data-option 14 --no-freeze --loss contrast --batch-size 16 --epochs 8
# Only semi-hard negative and hard negatives
python train.py --output-file contrast_nf_13_8e.pt --data-option 13 --no-freeze --loss contrast --batch-size 16 --epochs 8

# all
python train.py --output-file contrast_nf_1234_lr2e-6_8e.pt --data-option 1234 --no-freeze --loss contrast --batch-size 16 --epochs 8 --lr 2e-6
# No easy negatives
python train.py --output-file contrast_nf_134_lr2e-6_8e.pt --data-option 134 --no-freeze --loss contrast --batch-size 16 --epochs 8 --lr 2e-6
# No triplets with both random pos+neg
python train.py --output-file contrast_nf_123_lr2e-6_8e.pt --data-option 123 --no-freeze --loss contrast --batch-size 16 --epochs 8 --lr 2e-6
# No hard negatives
python train.py --output-file contrast_nf_14_lr2e-6_8e.pt --data-option 14 --no-freeze --loss contrast --batch-size 16 --epochs 8 --lr 2e-6
# Only semi-hard negative and hard negatives
python train.py --output-file contrast_nf_13_lr2e-6_8e.pt --data-option 13 --no-freeze --loss contrast --batch-size 16 --epochs 8 --lr 2e-6
#!/bin/sh

# all
python train.py --output-file contrast_1234_8e.pt --data-option 1234 --no-freeze --loss contrast --batch-size 16 --epochs 1
# No easy negatives
python train.py --output-file contrast_134_8e.pt --data-option 134 --no-freeze --loss contrast --batch-size 16 --epochs 1
# No triplets with both random pos+neg
python train.py --output-file contrast_123_8e.pt --data-option 123 --no-freeze --loss contrast --batch-size 16 --epochs 1
# No hard negatives
python train.py --output-file contrast_14_8e.pt --data-option 14 --no-freeze --loss contrast --batch-size 16 --epochs 1
# Only semi-hard negative and hard negatives
python train.py --output-file contrast_13_8e.pt --data-option 13 --no-freeze --loss contrast --batch-size 16 --epochs 1

# all
python train.py --output-file contrast_1234_8e.pt --data-option 1234 --no-freeze --loss contrast --batch-size 16 --epochs 8
# No easy negatives
python train.py --output-file contrast_134_8e.pt --data-option 134 --no-freeze --loss contrast --batch-size 16 --epochs 8
# No triplets with both random pos+neg
python train.py --output-file contrast_123_8e.pt --data-option 123 --no-freeze --loss contrast --batch-size 16 --epochs 8
# No hard negatives
python train.py --output-file contrast_14_8e.pt --data-option 14 --no-freeze --loss contrast --batch-size 16 --epochs 8
# Only semi-hard negative and hard negatives
python train.py --output-file contrast_13_8e.pt --data-option 13 --no-freeze --loss contrast --batch-size 16 --epochs 8

# all
python train.py --output-file contrast_1234_8e.pt --data-option 1234 --no-freeze --loss contrast --batch-size 16 --epochs 8 --lr 2e-6
# No easy negatives
python train.py --output-file contrast_134_8e.pt --data-option 134 --no-freeze --loss contrast --batch-size 16 --epochs 8 --lr 2e-6
# No triplets with both random pos+neg
python train.py --output-file contrast_123_8e.pt --data-option 123 --no-freeze --loss contrast --batch-size 16 --epochs 8 --lr 2e-6
# No hard negatives
python train.py --output-file contrast_14_8e.pt --data-option 14 --no-freeze --loss contrast --batch-size 16 --epochs 8 --lr 2e-6
# Only semi-hard negative and hard negatives
python train.py --output-file contrast_13_8e.pt --data-option 13 --no-freeze --loss contrast --batch-size 16 --epochs 8 --lr 2e-6
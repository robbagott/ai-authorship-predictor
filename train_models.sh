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

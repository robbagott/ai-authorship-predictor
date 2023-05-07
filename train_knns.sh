#!/bin/sh

for f in *.pt; do
  python train_knn.py --model-file ${f%.*}.pt --output-file knns/${f%.*}-knn.pt --results-file results/${f%.*}-results.txt
done
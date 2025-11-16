#!/bin/bash
#BSUB -J predrnn_kth_train
#BSUB -W 60
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=30GB]"
#BSUB -q gpuv100  # or gpua100
#BSUB -cwd /zhome/66/2/168935/uni/deeplearning/predrnn-pytorch
#BSUB -o batch/kth/python_%J.out
#BSUB -e batch/kth/python_%J.err

# Initialize Python environment
cd kth_script
bash predrnn_v2_kth_test.sh
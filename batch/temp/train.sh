#!/bin/bash
#BSUB -J predrnn_temperature_train
#BSUB -W 10
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -q gpuv100  # or gpua100
#BSUB -cwd /zhome/66/2/168935/uni/deeplearning/predrnn-pytorch
#BSUB -o batch/python_%J.out
#BSUB -e batch/python_%J.err

# Initialize Python environment
cd temp_script
bash predrnn_v2_temperature_train.sh
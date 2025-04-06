#!/bin/bash

source  ~/.bashrc
conda activate select_mixup

date

CUDA_VISIBLE_DEVICES=0 python data_monitoring.py

python data_selection.py 

CUDA_VISIBLE_DEVICES=0 python main.py

date

rm -rf __pycache__/
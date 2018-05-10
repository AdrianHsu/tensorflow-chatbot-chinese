#!/bin/sh

cd evaluation
#export CUDA_VISIBLE_DEVICES=""
export CUDA_VISIBLE_DEVICES=0
python3 main.py /home/data/mlds_hw2_2_data/test_input.txt ../output.txt
cd ..

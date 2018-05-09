#!/bin/sh

# python v3.6.0 & tensorflow-gpu v1.6 should already be installed
#pip3 install --user numpy
#pip3 install --user pandas
#pip3 install --user tqdm
#pip3 install --user ansicolors #  coloring the output message

# this is my own default path
default_dir='/home/data/mlds_hw2_2_data'
TEST_DIR=${1:-$default_dir}

default_out='./output.txt'
OUTPUT_FILENAME=${2:-$default_out}

#export CUDA_VISIBLE_DEVICES=0 #""
export CUDA_VISIBLE_DEVICES=""

python3 model_seq2seq.py --load_saver=1 --data_dir=$TEST_DIR \
            --test_mode=1 --output_filename=$OUTPUT_FILENAME \
            --batch_size=100 --save_dir=save #--with_attention=0

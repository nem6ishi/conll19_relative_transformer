#!/bin/bash

mkdir ./data
cd ./data
bash ../preprocess/wmt2014_from_stanford_sp40k.sh
cd ..
DATA=./data/wmt2014/sentencepiece_40k


export CUDA_VISIBLE_DEVICES=0
MODEL_DIR=./model/test
SETTING_FILE=./setting/rnn_rel_trans.json
DATA=./data/wmt2014/sentencepiece_40k

mkdir -p $MODEL_DIR
echo "running on [SERVER: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"] >> $MODEL_DIR/log.txt
nohup python3 ./run_train.py \
  -s $SETTING_FILE \
  -d $MODEL_DIR \
  -vs $DATA/sp40k.vocab \
  -vt $DATA/sp40k.vocab \
  -ts $DATA/train.en \
  -tt $DATA/train.de \
  -ds $DATA/dev.en \
  -st $DATA/dev.de \
  >> $MODEL_DIR/log.txt &

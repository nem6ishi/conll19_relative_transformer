# WMT2014
DATA=./data/wmt2014/sentencepiece_40k
SRC_TEST=$DATA/test.en
TGT_TEST=$DATA/test.de

MODEL_DIR=./model/wmt2014
mkdir -p $MODEL_DIR/test

nohup python3 ./run_test.py \
  -d $MODEL_DIR \
  --src_test $SRC_TEST \
  --tgt_test $TGT_TEST \
  >> $MODEL_DIR/test/log.txt &

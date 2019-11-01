# WMT2014
DATA=./data/wmt2014/sentencepiece_40k

MODEL_DIR=./model/wmt2014
SETTING_FILE=./setting/rnn_rel_trans.json

mkdir -p $MODEL_DIR
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

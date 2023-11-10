# Transformer with Relative Position

This is the official code for the paper **On the Relation between Position Information and Sentence Length in Neural Machine Translation** accepted by the 23rd Conference on Computational Natural Language Learning (CoNLL 2019).

The Code includes 5(+1) kinds of neural machine translation models;
- RNN-based model [Luong+, 2015]
- Transformer [Vaswani+, 2017]
- Rel-Transformer [Shaw+, 2018]
- RNN-Transformer [Our proposed]
- RR-Transformer [Shaw+, 2018]+[Our proposed]
- ( Transformer [Vaswani+, 2017] (position embeddings ver.) )

please refer the proceeding paper for details of the models.

## Requirements
- Python 3
- Pytorch ver. 0.4.1
- numpy
- tensorboardX

- SentencePiece: Not for NMT system but for preprocess part. If you try get started code, please install this in advance. (https://github.com/google/sentencepiece)
- Kytea: For both prepcoress part and evaluation part. Please install if you try ASPEC dataset, and set setting file as ["options"]["use_kytea"]:true.(http://www.phontron.com/kytea/)


## Get started
### Install
```
git clone https://github.com/nem6ishi/conll19_relative_transformer.git
cd conll19_relative_transformer
pip install -e .
```

### Data preprocess

- WMT2014 En-De (SentencePiece required)
```
mkdir -p ./data
cd ./data
bash ../preprocess/wmt2014_from_stanford_sp40k.sh
cd ..
```


- ASPEC En-Ja (SentencePiece & Kytea required)
    - Get data from http://orchid.kuee.kyoto-u.ac.jp/ASPEC/ .
    - Overwrite the first line of `./preprocess/aspec_sp16k.sh`:  `ASPEC_DIR=/path/to/ASPEC/original`
```
mkdir -p ./data
cd ./data
bash ../preprocess/aspec_sp16k.sh
cd ..
```

### Training
```
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
  -dt $DATA/dev.de \
  >> $MODEL_DIR/log.txt &
```
- By customizing setting file, you can run the code above only with `-s` option.
- You can check the training process with tensorboard.


### Test
```
DATA=./data/wmt2014/sentencepiece_40k
SRC_TEST=$DATA/test.en
TGT_TEST=$DATA/test.de

MODEL_DIR=./model/wmt2014
mkdir -p ${MODEL_DIR}/test

nohup python3 ./run_test.py \
  -d $MODEL_DIR \
  --src_test $SRC_TEST \
  --tgt_test $TGT_TEST \
  >> ${MODEL_DIR}/test/log.txt &
```
- The output translation is saved under `${MODEL_DIR}/test`.


### Settings for each NMT models
`./setting/` includes 5(+1) setting files for each NMT models.
There are some slight differences in the settings for the model architecture.
Please use the setting file of the model you want to try for the `-s` option in training command.
- rnn_nmt.json:

## Reference

```
@inproceedings{neishi-yoshinaga-2019-relation,
    title = "On the Relation between Position Information and Sentence Length in Neural Machine Translation",
    author = "Neishi, Masato  and
      Yoshinaga, Naoki",
    editor = "Bansal, Mohit  and
      Villavicencio, Aline",
    booktitle = "Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/K19-1031",
    doi = "10.18653/v1/K19-1031",
    pages = "328--338",
```

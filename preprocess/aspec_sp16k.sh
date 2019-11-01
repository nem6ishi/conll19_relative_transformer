ASPEC_DIR=/path/to/ASPEC/original


CD=$(pwd)
if [ ! -d mosesdecoder_RELEASE-2.1.1 ]; then
  git clone https://github.com/moses-smt/mosesdecoder.git -b RELEASE-2.1.1 --depth 1 mosesdecoder_RELEASE-2.1.1
fi
MOSES_SCRIPT=$CD/mosesdecoder_RELEASE-2.1.1/scripts

#####ASPEC#####
DIR=$(pwd)/aspec
mkdir -p $DIR

echo "start: extraction"
mkdir -p $DIR/original
#extraction
for name in dev test; do
  perl -ne 'chomp; @a=split/ \|\|\| /; print $a[2], "\n";' < ${ASPEC_DIR}/${name}/${name}.txt > $DIR/original/${name}.ja &
  perl -ne 'chomp; @a=split/ \|\|\| /; print $a[3], "\n";' < ${ASPEC_DIR}/${name}/${name}.txt > $DIR/original/${name}.en &
done
for name in train-1 train-2 train-3; do
  perl -ne 'chomp; @a=split/ \|\|\| /; print $a[3], "\n";' < ${ASPEC_DIR}/train/${name}.txt > $DIR/original/${name}.ja &
  perl -ne 'chomp; @a=split/ \|\|\| /; print $a[4], "\n";' < ${ASPEC_DIR}/train//${name}.txt > $DIR/original/${name}.en &
done
wait

for lang in ja en; do
  cat $DIR/original/train-{1,2,3}.${lang} >  $DIR/original/train.${lang} &
done
wait
rm $DIR/original/train-*

echo "done: extraction"


echo "start: process"
mkdir -p $DIR/processed
### JAPANESE
#tokenization
for file in train dev test; do
  cat $DIR/original/${file}.ja | \
  kytea -out tok > $DIR/processed/${file}.ja &
done


### ENGLISH
#tokenization
for file in train dev test; do
  cat $DIR/original/${file}.en | \
  perl ${MOSES_SCRIPT}/tokenizer/tokenizer.perl -l en -no-escape\
  > $DIR/processed/${file}.tokenized.en &
done
wait


#train truecase model
perl ${MOSES_SCRIPT}/recaser/train-truecaser.perl \
  --model $DIR/processed/truecase.model --corpus $DIR/processed/train.tokenized.en

#truecasing
for file in train dev test; do
  perl ${MOSES_SCRIPT}/recaser/truecase.perl \
    --model $DIR/processed/truecase.model < $DIR/processed/${file}.tokenized.en \
    > $DIR/processed/${file}.en &
done
wait

echo "done: process"


echo "start: sentencepiece"
mkdir -p $DIR/sentencepieced
### SENTENCE_PIECE
cat $DIR/processed/train.{en,ja} > $DIR/sentencepieced/train_for_sp.all
spm_train --input=$DIR/sentencepieced/train_for_sp.all --model_prefix=$DIR/sentencepieced/sp16k  --vocab_size=16000 --character_coverage=1.0
rm $DIR/sentencepieced/train_for_sp.all

for file in train dev test; do
  spm_encode --model=$DIR/sentencepieced/sp16k.model < $DIR/processed/${file}.en > $DIR/sentencepieced/${file}.en &
done
for file in train dev test; do
  spm_encode --model=$DIR/sentencepieced/sp16k.model < $DIR/processed/${file}.ja > $DIR/sentencepieced/${file}.ja &
done
wait
echo "done: sentencepiece"

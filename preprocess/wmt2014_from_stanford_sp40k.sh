DIR=$(pwd)/wmt2014
orig=$DIR/orig
size=40
sentencepiece=$DIR/sentencepiece_${size}k
src=en
tgt=de


mkdir -p $orig
cd $orig

wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en &
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de &

wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.en
mv newstest2013.en dev.en
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.de
mv newstest2013.de dev.de

wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en
mv newstest2014.en test.en
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de
mv newstest2014.de test.de
wait
cd $DIR


echo "start: sentencepiece"
mkdir -p $sentencepiece
### SENTENCE_PIECE
cat $orig/train.{$src,$tgt} > $sentencepiece/train_for_sp.all
spm_train --input=$sentencepiece/train_for_sp.all --model_prefix=$sentencepiece/sp${size}k  --vocab_size=${size}000 --character_coverage=1.0
rm $sentencepiece/train_for_sp.all

for l in $src $tgt; do
  for file in train dev test; do
    spm_encode --model=$sentencepiece/sp${size}k.model < $orig/${file}.$l > $sentencepiece/${file}.$l &
  done
done
wait
echo "done: sentencepiece"

cd ${DIR}/..

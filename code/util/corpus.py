import torch, copy, os, numpy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SingleCorpus:
  def __init__(self, lang, file_path, max_length=0, num_imort_line=-1): ### set max_length less than or equal to 0 to disable it
    if not os.path.isfile(file_path):
      raise ValueError("File does not exist: {0}".format(file_path))

    self.lang = lang
    self.file_path = file_path
    self.max_length = max_length # the limit at first, but will be updated as "actual" max length during import
    self.num_imort_line = num_imort_line

    self.corpus_size = 0
    self.sentences = {}
    self.lengths = {}
    self.is_not_converted = {}

    self.remove_indexes_set = self.check_length()


  def check_length(self): # check length before importing. do this before import for parallel corpus
    remove_indexes_set = set()
    if self.max_length > 0:
      with open(self.file_path, "r", encoding='utf-8') as file:
        for i, sent in enumerate(file):
          assert "　" not in sent
          if self.num_imort_line>0 and i > self.num_imort_line:
            break
          len_sent = sent.count(' ') + 1
          if len_sent > self.max_length:
            remove_indexes_set.add(i)
      self.corpus_size = i + 1
    else:
      self.corpus_size = sum(1 for line in open(self.file_path))
    return remove_indexes_set


  def import_file(self):
    self.corpus_size = 0 # reset corpus_size
    tmp_max_length = 0
    with open(self.file_path, "r", encoding='utf-8') as file:
      for i, sent in enumerate(file):
        if self.num_imort_line>0 and i > self.num_imort_line:
          break
        if i not in self.remove_indexes_set:
          self.sentences[self.corpus_size] = sent ### save sentence as str first. convert it later
          length = sent.count(' ') + 1 + 2 ### '2' for "SEQUENCE_START" and "SEQUENCE_END"
          self.lengths[self.corpus_size] = length
          tmp_max_length = max([tmp_max_length, length])
          self.is_not_converted[self.corpus_size] = True
          self.corpus_size += 1
        else:
          self.remove_indexes_set.remove(i)
    self.max_length = tmp_max_length # update to actual max_length


  def convert_into_indexes(self, sentence_index_list):
    for idx in sentence_index_list:
      if self.is_not_converted[idx]:
        self.sentences[idx] = self.lang.sentence2indexes(self.sentences[idx].split())
        self.is_not_converted[idx] = False



class ParallelCorpus:
  def __init__(self, src_lang, tgt_lang, src_file_path, tgt_file_path, max_length=0, num_imort_line=-1):
    self.corpus_size = 0
    self.share_corpus = True if (src_lang.path == tgt_lang.path and src_file_path == tgt_file_path) else False
    self.max_length = 0 # this will be updated when importing
    self.num_imort_line = num_imort_line

    self.src_lang = src_lang
    self.tgt_lang = tgt_lang

    self.src_corpus = SingleCorpus(src_lang, src_file_path, max_length, self.num_imort_line)
    if self.share_corpus or tgt_file_path == None : # None for test time in which there is no tgt_corpus
      self.tgt_corpus = self.src_corpus
    else:
      self.tgt_corpus = SingleCorpus(tgt_lang, tgt_file_path, max_length, self.num_imort_line)

    if not self.share_corpus:
      assert self.src_corpus.corpus_size == self.tgt_corpus.corpus_size
      combined_remove_indexes_set = self.src_corpus.remove_indexes_set | self.tgt_corpus.remove_indexes_set
      self.src_corpus.remove_indexes_set = copy.deepcopy(combined_remove_indexes_set)
      self.tgt_corpus.remove_indexes_set = copy.deepcopy(combined_remove_indexes_set)


  def import_file(self):
    self.src_corpus.import_file()
    if not self.share_corpus:
      self.tgt_corpus.import_file()
    assert self.src_corpus.corpus_size==self.tgt_corpus.corpus_size
    self.corpus_size = self.src_corpus.corpus_size
    self.max_length = max([self.src_corpus.max_length, self.tgt_corpus.max_length])


  def convert_into_indexes(self, sentence_index_list):
    self.src_corpus.convert_into_indexes(sentence_index_list)
    if not self.share_corpus:
      self.tgt_corpus.convert_into_indexes(sentence_index_list)



class SingleBatch:
  def __init__(self, corpus, fixed_batch_size, num_cat_sent=1):
    self.fixed_batch_size = fixed_batch_size
    self.num_cat_sent = num_cat_sent # param to concatenate 2 or more sentences as one
    self.enable_cat = False
    self.batch_size = 0
    self.corpus = corpus
    self.sample_idxs = []
    self.sentences = []
    self.lengths = []
    self.masks = []


  def generate_random_indexes(self):
    self.enable_cat = True if self.num_cat_sent > 1 else False
    self.sample_idxs = numpy.random.randint(0, self.corpus.corpus_size, self.fixed_batch_size*self.num_cat_sent)


  def generate_sequential_indexes(self, num_iter):
    self.enable_cat = False
    num_done = self.fixed_batch_size * num_iter
    num_rest = self.corpus.corpus_size - num_done
    if num_rest < self.fixed_batch_size:
      batch_size = num_rest
    else:
      batch_size = self.fixed_batch_size
    self.sample_idxs = list(range(num_done, num_done+batch_size))


  def generate_batch(self, reverse=False):
    self.sentences = []
    self.lengths = []
    self.batch_size = len(self.sample_idxs)
    self.corpus.convert_into_indexes(self.sample_idxs)

    sent = []
    length = 0
    for i, idx in enumerate(self.sample_idxs):
      sent += copy.deepcopy(self.corpus.sentences[idx])
      length += self.corpus.lengths[idx]

      if (i+1)%self.num_cat_sent==0 or not self.enable_cat:
        if reverse:
          sent.reverse()
        self.sentences.append(torch.LongTensor(sent).to(device))
        self.lengths.append(length)
        sent = []
        length = 0

    self.sentences = torch.nn.utils.rnn.pad_sequence(self.sentences,
                                                     batch_first=True,
                                                     padding_value=self.corpus.lang.vocab2index["PADDING"])
    self.masks = (self.sentences == torch.zeros(self.sentences.size(), dtype=torch.long).to(device))



class ParallelBatch:
  def __init__(self, parallel_corpus, fixed_batch_size, num_cat_sent=1):
    self.fixed_batch_size = fixed_batch_size
    self.src_corpus = parallel_corpus.src_corpus
    self.tgt_corpus = parallel_corpus.tgt_corpus
    self.batch_size = 0

    self.src_batch = SingleBatch(self.src_corpus, self.fixed_batch_size, num_cat_sent)
    self.tgt_batch = SingleBatch(self.tgt_corpus, self.fixed_batch_size, num_cat_sent)


  def generate_random_indexes(self):
    self.src_batch.generate_random_indexes()
    self.tgt_batch.sample_idxs = self.src_batch.sample_idxs
    self.tgt_batch.enable_cat = self.src_batch.enable_cat


  def generate_sequential_indexes(self, num_iter):
    self.src_batch.generate_sequential_indexes(num_iter)
    self.tgt_batch.sample_idxs = self.src_batch.sample_idxs
    self.tgt_batch.enable_cat = self.src_batch.enable_cat

  def generate_batch(self, src_reverse=False, tgt_reverse=False):
    self.src_batch.generate_batch(src_reverse)
    self.tgt_batch.generate_batch(tgt_reverse)
    self.batch_size = self.src_batch.batch_size

import os



#special_vocabs = ["PADDING", "MASK", "UNK", "SEQUENCE_START", "SEQUENCE_END"]
special_vocabs = ["PADDING", "UNK", "SEQUENCE_START", "SEQUENCE_END"]

class Lang:
  def __init__(self, path):
    if not os.path.isfile(path):
      raise ValueError("File does not exist: {}".format(path))
    self.path = path
    self.vocab2index = {}
    self.index2vocab = {}
    self.vocab_size = 0
    self.create_vocab()


  def add_word(self, word):
    if word not in self.vocab2index and len(word) > 0:
      self.vocab2index[word] = self.vocab_size
      self.index2vocab[self.vocab_size] = word
      self.vocab_size += 1


  def create_vocab(self):
    for word in special_vocabs:
      self.add_word(word)
    with open(self.path, "r", encoding='utf-8') as file:
      for line in file:
        word = line.split()[0]
        self.add_word(word)


  def sentence2indexes(self, sentence_as_list, reverse=False):
    index_list = [self.vocab2index["SEQUENCE_START"]]
    for each in sentence_as_list:
      index_list.append(self.vocab2index[each] if each in self.vocab2index else self.vocab2index["UNK"])
    index_list.append(self.vocab2index["SEQUENCE_END"])

    if reverse:
      index_list.reverse()

    return index_list


  def indexes2sentence(self, index_list, clean=True, reverse=False):
    sentence_as_list = []
    index_list = list(index_list)

    if reverse:
      index_list.reverse()

    for word_index in index_list:
      word_index = int(word_index)
      if word_index not in self.index2vocab:
        raise ValueError("Vocab index does not exist: {}".format(word_index))
      word = self.index2vocab[word_index]
      if word != "PADDING" or clean == False:
        sentence_as_list.append(word)

    if clean:
      if sentence_as_list[0] == "SEQUENCE_START":
        sentence_as_list.pop(0)
      if len(sentence_as_list) > 0 and sentence_as_list[-1] == "SEQUENCE_END":
        sentence_as_list.pop()

    return sentence_as_list

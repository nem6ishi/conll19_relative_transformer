import torch, random

from code.model.module import rnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Seq2SeqModel(torch.nn.Module):
  def __init__(self, setting, src_lang, tgt_lang):
    super(Seq2SeqModel, self).__init__()
    self.src_lang, self.tgt_lang = src_lang, tgt_lang
    self.encoder = rnn.EncoderRNN(
                     src_lang.vocab_size,
                     setting["encoder_vars"]["emb_dim"],
                     setting["encoder_vars"]["model_dim"],
                     setting["encoder_vars"]["num_layers"],
                     setting["encoder_vars"]["bi_directional"],
                     setting["decoder_vars"]["model_dim"],
                     setting["decoder_vars"]["num_layers"],
                     dropout_p=setting["train_vars"]["dropout_p"],
                     padding_idx=src_lang.vocab2index["PADDING"])

    num_direction = 2 if setting["encoder_vars"]["bi_directional"] else 1
    self.attention_type = setting["decoder_vars"]["attention_type"]
    if self.attention_type == "luong_general":
      self.decoder = rnn.LuongDecoderRNN(
                       src_lang.vocab_size,
                       setting["decoder_vars"]["emb_dim"],
                       setting["decoder_vars"]["model_dim"],
                       setting["decoder_vars"]["num_layers"],
                       setting["encoder_vars"]["model_dim"] * num_direction,
                       dropout_p=setting["train_vars"]["dropout_p"],
                       padding_idx=tgt_lang.vocab2index["PADDING"])
    else:
      raise NotImplementedError()

    if setting["options"]["share_embedding"]:
      assert setting["paths"]["src_vocab"] == setting["paths"]["tgt_vocab"]
      self.decoder.embedding = self.encoder.embedding



  def translate_for_train(self, batch):
    outputs, (hidden, cell) = self.encoder(batch.src_batch)
    word_outputs, prob_outputs = self.decode(batch,
                                             outputs,
                                             hidden,
                                             cell,
                                             batch.tgt_batch.sentences.size(1),
                                             force_teaching_p=1.0)
    return prob_outputs[:, 1:]



  def translate(self, batch, max_length):
    outputs, (hidden, cell) = self.encoder(batch.src_batch)
    word_outputs, prob_outputs = self.decode(batch,
                                             outputs,
                                             hidden,
                                             cell,
                                             max_length)
    return word_outputs, prob_outputs



  def decode(self, para_batch, outputs, hidden, cell, max_target_length, force_teaching_p=-1):
    batch_size = para_batch.batch_size
    flag = [False for i in range(batch_size)]

    decoder_word_outputs = torch.zeros((batch_size, max_target_length), dtype=torch.long).to(device)
    decoder_word_outputs[:, 0] = self.tgt_lang.vocab2index["SEQUENCE_START"] # first word is always SEQUENCE_START
    decoder_prob_outputs = torch.zeros((batch_size, max_target_length, self.tgt_lang.vocab_size), dtype=torch.float).to(device)
    dec_state = rnn.DecoderState(self.attention_type, hidden, cell)

    # decode words one by one
    for i in range(max_target_length-1):
      if force_teaching_p >= random.random():
        decoder_input = para_batch.tgt_batch.sentences[:, i].unsqueeze(1)
      else:
        decoder_input = decoder_word_outputs[:, i].unsqueeze(1)

      decoder_output = self.decoder(decoder_input, dec_state, outputs, para_batch.src_batch.masks)
      likelihood, index = decoder_output.data.topk(1)
      index = index.squeeze(1)

      for j, each in enumerate(index):
        if flag[j]:
          index[j] = self.tgt_lang.vocab2index["PADDING"]
        elif int(each) == self.tgt_lang.vocab2index["SEQUENCE_END"]:
          flag[j] = True

      decoder_word_outputs[:, i+1] = index
      decoder_prob_outputs[:, i+1] = decoder_output

      if force_teaching_p == -1 and all(flag):
        break

    return decoder_word_outputs, decoder_prob_outputs

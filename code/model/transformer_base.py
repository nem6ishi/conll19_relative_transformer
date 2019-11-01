import torch

from code.model.module import transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class TransformerBaseModel(torch.nn.Module):

  def translate_for_train(self, batch):
    outputs = self.encoder(batch.src_batch.sentences)
    prob_outputs = self.decode_for_train(batch.tgt_batch.sentences[:, :-1],
                                            outputs,
                                            batch.src_batch.masks)
    return prob_outputs

  def translate(self, batch, max_length):
    outputs = self.encoder(batch.src_batch.sentences)
    word_outputs, prob_outputs = self.decode(outputs,
                                             batch.src_batch.masks,
                                             max_length)
    return word_outputs, prob_outputs

  def decode_for_train(self, tgt_sent, encoder_outputs, src_mask):
    decoder_output = self.decoder(tgt_sent, encoder_outputs, src_mask)
    decoder_prob_outputs = self.generator(decoder_output)
    return decoder_prob_outputs

  def decode(self, encoder_outputs, src_mask, max_target_length):
    start_token = self.tgt_lang.vocab2index["SEQUENCE_START"]
    end_token = self.tgt_lang.vocab2index["SEQUENCE_END"]

    batch_size = encoder_outputs.size(0)
    flag = [False for i in range(batch_size)]
    decoder_word_outputs = torch.zeros((batch_size, max_target_length), dtype=torch.long).to(device)
    decoder_word_outputs[:, 0] = start_token # first word
    decoder_prob_outputs = torch.zeros((batch_size, max_target_length, self.tgt_lang.vocab_size), dtype=torch.float).to(device)

    layer_cache = {}
    for i in range(self.decoder.num_layers):
      layer_cache[i] = {"kv_mask": None, "key": None, "value": None}

    # decode words one by one
    for i in range(1, max_target_length):
      decoder_output = self.decoder(decoder_word_outputs[:, i-1:i], encoder_outputs, src_mask, i-1, layer_cache)
      generator_output = self.generator(decoder_output[:, -1])

      likelihood, index = generator_output.data.topk(1)
      index = index.squeeze(1)

      for j, each in enumerate(index):
        if flag[j]:
          index[j] = self.tgt_lang.vocab2index["PADDING"]
        elif int(each) == end_token:
          flag[j] = True

      decoder_word_outputs[:, i] = index
      decoder_prob_outputs[:, i] = generator_output

      if all(flag):
        break

    return decoder_word_outputs, decoder_prob_outputs




class RNNTransformerBaseModel(TransformerBaseModel):

  def decode_for_train(self, tgt_sent, encoder_outputs, src_mask):
    decoder_output, _hidden = self.decoder(tgt_sent, encoder_outputs, src_mask)
    decoder_prob_outputs = self.generator(decoder_output)
    return decoder_prob_outputs

  def decode():
    raise NotImplementedError()

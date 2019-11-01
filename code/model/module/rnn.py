import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class EncoderRNN(torch.nn.Module):
  def __init__(self, vocab_size, emb_dim, model_dim, num_layers, bi_directional, decoder_model_dim, decoder_num_layers, dropout_p=0.1, padding_idx=0, reverse_input=False):
    super(EncoderRNN, self).__init__()

    self.vocab_size = vocab_size
    self.emb_dim = emb_dim
    self.model_dim = model_dim
    self.num_layers = num_layers
    self.bi_directional = bi_directional
    self.n_directions = 2 if self.bi_directional else 1

    self.decoder_num_layers = decoder_num_layers
    self.decoder_model_dim = decoder_model_dim

    self.dropout_p = dropout_p
    self.reverse_input = reverse_input

    self.embedding = torch.nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=padding_idx)
    self.lstm = torch.nn.LSTM(self.emb_dim, self.model_dim, self.num_layers, batch_first=True, dropout=self.dropout_p, bidirectional=self.bi_directional)
    self.dropout = torch.nn.Dropout(self.dropout_p)

    if (not self.bi_directional) and (self.num_layers == self.decoder_num_layers) and (self.model_dim == self.decoder_model_dim):
      self.need_transform = False
    else:
      self.need_transform = True
      self.linear_hidden = torch.nn.Linear(self.num_layers*self.model_dim*self.n_directions, self.decoder_num_layers*self.decoder_model_dim)
      self.linear_cell = torch.nn.Linear(self.num_layers*self.model_dim*self.n_directions, self.decoder_num_layers*self.decoder_model_dim)


  def forward(self, batch):
    sort_indexes = self.sort(batch)

    if self.reverse_input:
      sentences = self.reverse_sentence(batch)
    else:
      sentences = batch.sentences

    embedded = self.dropout(self.embedding(sentences)) # [B,T,H]
    embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, batch.lengths, batch_first=True) # form matrix into list, to care paddings in rnn
    outputs, (hidden, cell) = self.lstm(embedded)
    outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True) # reform into matrix
    (hidden, cell) = self.linear_transform_for_decoder_hidden(hidden, cell)

    outputs, hidden, cell = self.resort(sort_indexes, batch, outputs, hidden, cell)

    return outputs, (hidden, cell)


  def linear_transform_for_decoder_hidden(self, hidden, cell):
    if not self.need_transform:
      return (hidden, cell)

    batch_size = hidden.size(1)

    hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1)
    hidden = torch.tanh(self.linear_hidden(hidden))
    hidden = hidden.view(batch_size, self.decoder_num_layers, self.decoder_model_dim).transpose(0, 1).contiguous()

    cell = cell.transpose(0, 1).contiguous().view(batch_size, -1)
    cell = torch.tanh(self.linear_cell(cell))
    cell = cell.view(batch_size, self.decoder_num_layers, self.decoder_model_dim).transpose(0, 1).contiguous()

    return (hidden, cell)


  def sort(self, batch):
    batch.lengths, sort_indexes = torch.LongTensor(batch.lengths).sort(0, descending=True)
    batch.lengths = batch.lengths.tolist()
    batch.sentences = batch.sentences[sort_indexes]
    batch.masks = batch.masks[sort_indexes]
    return sort_indexes


  def resort(self, sort_indexes, batch, outputs, hidden, cell):
    _, resort_indexes = sort_indexes.sort(0, descending=False)

    batch.lengths = torch.LongTensor(batch.lengths)[resort_indexes].tolist()
    batch.sentences = batch.sentences[resort_indexes]
    batch.masks = batch.masks[resort_indexes]

    outputs = outputs[resort_indexes]
    hidden = hidden[:, resort_indexes]
    cell = cell[:, resort_indexes]

    return outputs, hidden, cell


  def reverse_sentence(self, batch):
    batch_length = batch.sentences.size(1)
    reversed_sentences = torch.LongTensor(batch.sentences.size()).to(device) # keep original sentences from reversing

    # turn all
    reversed_sentences = batch.sentences[:, list(range(batch_length-1, -1, -1))]

    # shift words and paddings
    for i, each in enumerate(batch.lengths):
      num_padding = batch_length - each
      if num_padding > 0:
        reversed_sentences[i] = torch.cat((reversed_sentences[i, num_padding:], reversed_sentences[i, :num_padding]), 0)

    return reversed_sentences





class LuongDecoderRNN(torch.nn.Module):
  def __init__(self, vocab_size, emb_dim, model_dim, num_layers, encoder_output_dim, dropout_p=0.1, padding_idx=0):
    super(LuongDecoderRNN, self).__init__()

    self.vocab_size = vocab_size
    self.emb_dim = emb_dim
    self.model_dim = model_dim
    self.num_layers = num_layers
    self.dropout_p = dropout_p

    self.encoder_output_dim = encoder_output_dim

    self.embedding = torch.nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=padding_idx)
    self.lstm = torch.nn.LSTM(self.emb_dim+self.model_dim, self.model_dim, num_layers=self.num_layers, batch_first=True, dropout=self.dropout_p)
    self.Wa = torch.nn.Linear(self.encoder_output_dim, self.num_layers*self.model_dim, bias=False)
    self.Wc = torch.nn.Linear(self.num_layers*self.model_dim + self.encoder_output_dim, self.model_dim, bias=False)
    self.Ws = torch.nn.Linear(self.model_dim, self.vocab_size)
    self.dropout = torch.nn.Dropout(self.dropout_p)


  def forward(self, input, dec_state, encoder_outputs, src_mask):
    batch_size = input.size(0)

    embedded = self.dropout(self.embedding(input))
    lstm_input = torch.cat((embedded, dec_state.h_tilde.unsqueeze(1)), 2) # input feeding

    output, (dec_state.hidden, dec_state.cell) = self.lstm(lstm_input, (dec_state.hidden, dec_state.cell))
    contexts, dec_state.att_weights = self.attention(dec_state.hidden, encoder_outputs, src_mask)

    dec_state.h_tilde = torch.tanh(self.Wc(torch.cat((dec_state.hidden.transpose(0, 1).contiguous().view(batch_size, -1), contexts), 1)))
    output = self.Ws(dec_state.h_tilde)
    output = torch.nn.functional.log_softmax(output, dim=1).squeeze(1)

    return output


  def attention(self, hidden, encoder_outputs, mask):
    if not torch.is_tensor(encoder_outputs): # not use attention
      contexts = torch.zeros(batch_size, self.encoder_output_dim).to(device)
      return contexts, None

    batch_size = encoder_outputs.size(0)
    input_sentence_length = encoder_outputs.size(1)

    hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1)
    score = self.Wa(encoder_outputs)
    score = torch.bmm(hidden.unsqueeze(1), score.transpose(1,2)).squeeze(1)
    score.data.masked_fill_(mask, float('-inf'))

    att_weights = torch.nn.functional.softmax(score, dim=1)
    contexts = torch.bmm(att_weights.unsqueeze(1), encoder_outputs).squeeze(1)

    return contexts, att_weights



class DecoderState:
  def __init__(self, attention_type, hidden, cell=None):
    self.hidden = hidden
    self.cell = cell
    if attention_type == "luong_general":
      size = (hidden.size(1), hidden.size(2))
      self.h_tilde = torch.zeros(size).to(device)
    self.att_weights = None

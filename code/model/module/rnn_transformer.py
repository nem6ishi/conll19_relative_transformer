import torch

from code.model.module import transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class RNNTransformerEncoder(torch.nn.Module):
  def __init__(self, vocab_size, emb_dim, model_dim, ff_dim, num_layers, num_head, bi_directional, dropout_p=0.1, padding_idx=0):
    super(RNNTransformerEncoder, self).__init__()

    self.vocab_size = vocab_size
    self.emb_dim = emb_dim
    self.model_dim = model_dim
    self.ff_dim = ff_dim
    self.num_layers = num_layers
    self.num_head = num_head
    self.bi_directional = bi_directional
    self.num_directions = 2 if bi_directional else 1

    self.dropout_p = dropout_p

    self.embedding = torch.nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=padding_idx)
    self.rnn = torch.nn.GRU(self.emb_dim, self.model_dim, num_layers=1, batch_first=True, bidirectional=self.bi_directional)

    if self.bi_directional:
      self.ff = torch.nn.Linear(self.model_dim*2, self.model_dim)

    self.layers = torch.nn.ModuleList([transformer.TransformerEncoderLayer(self.model_dim, self.num_head, self.ff_dim, self.dropout_p) for i in range(self.num_layers)])
    self.dropout = torch.nn.Dropout(self.dropout_p)


  def forward(self, input):
    mask = (input==torch.zeros(input.size(), dtype=torch.long).to(device))
    embedded = self.dropout(self.embedding(input)) # [B,T,H]

    x, _hidden = self.rnn(embedded)
    if self.bi_directional:
      x = self.ff(x)
    mask_here = mask.unsqueeze(-1).expand(-1, -1, x.size(-1))
    x = x.masked_fill(mask_here, 0.0)

    for layer in self.layers:
      x = layer(x, mask)

    return x



class RNNTransformerDecoder(torch.nn.Module):
  def __init__(self, vocab_size, emb_dim, model_dim, ff_dim, num_layers, num_head, dropout_p=0.1, padding_idx=0):
    super(RNNTransformerDecoder, self).__init__()

    self.vocab_size = vocab_size
    self.emb_dim = emb_dim
    self.model_dim =  model_dim
    self.ff_dim = ff_dim
    self.num_layers = num_layers
    self.num_head = num_head

    self.dropout_p = dropout_p

    self.embedding = torch.nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=padding_idx)
    self.rnn = torch.nn.GRU(self.emb_dim, self.model_dim, num_layers=1, batch_first=True, bidirectional=False)
    self.layers = torch.nn.ModuleList([transformer.TransformerDecoderLayer(self.model_dim, self.num_head, self.ff_dim, self.dropout_p) for i in range(self.num_layers)])
    self.dropout = torch.nn.Dropout(self.dropout_p)


  def forward(self, input, encoder_output, src_mask, layer_cache=None, hidden=None):
    tgt_mask = (input == torch.zeros(input.size(), dtype=torch.long).to(device))
    embedded = self.dropout(self.embedding(input))

    if torch.is_tensor(hidden):
      x, hidden = self.rnn(embedded, hidden)
    else:
      x, hidden = self.rnn(embedded)

    mask_here = tgt_mask.unsqueeze(-1).expand(-1, -1, x.size(-1))
    x = x.masked_fill(mask_here, 0.0)

    for i, layer in enumerate(self.layers):
      x = layer(x, encoder_output, src_mask, tgt_mask,
                layer_cache=layer_cache[i] if layer_cache!=None else None)
    return x, hidden

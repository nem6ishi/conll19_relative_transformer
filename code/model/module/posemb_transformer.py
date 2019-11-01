import torch

from code.model.module import transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class PosEmbTransformerEncoder(torch.nn.Module):
  def __init__(self, vocab_size, emb_dim, model_dim, ff_dim, num_layers, num_head, dropout_p=0.1, padding_idx=0):
    super(PosEmbTransformerEncoder, self).__init__()

    self.vocab_size = vocab_size
    self.emb_dim = emb_dim
    self.model_dim =  model_dim
    self.ff_dim = ff_dim
    self.num_layers = num_layers
    self.num_head = num_head

    self.dropout_p = dropout_p

    self.embedding = torch.nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=padding_idx)
    max_sent_length = 1000
    self.pos_embedding = torch.nn.Embedding(max_sent_length, self.emb_dim)
    self.layers = torch.nn.ModuleList([transformer.TransformerEncoderLayer(self.model_dim, self.num_head, self.ff_dim, self.dropout_p) for i in range(self.num_layers)])
    self.dropout = torch.nn.Dropout(self.dropout_p)


  def forward(self, input):
    mask = (input==torch.zeros(input.size(), dtype=torch.long).to(device))
    embedded = self.dropout(self.embedding(input)) # [B,T,H]

    pos_id = torch.LongTensor([list(range(input.size(1))) for i in range(input.size(0))]).to(device)
    pos_emb = self.pos_embedding(pos_id)
    x = self.dropout(embedded + pos_emb)
    x = x.masked_fill(mask.unsqueeze(-1).expand(-1, -1, x.size(-1)), 0.0)

    for layer in self.layers:
      x = layer(x, mask)
    return x



class PosEmbTransformerDecoder(torch.nn.Module):
  def __init__(self, vocab_size, emb_dim, model_dim, ff_dim, num_layers, num_head, dropout_p=0.1, padding_idx=0):
    super(PosEmbTransformerDecoder, self).__init__()

    self.vocab_size = vocab_size
    self.emb_dim = emb_dim
    self.model_dim =  model_dim
    self.ff_dim = ff_dim
    self.num_layers = num_layers
    self.num_head = num_head

    self.dropout_p = dropout_p

    self.embedding = torch.nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=padding_idx)
    max_sent_length = 1000
    self.pos_embedding = torch.nn.Embedding(max_sent_length, self.emb_dim)
    self.layers = torch.nn.ModuleList([transformer.TransformerDecoderLayer(self.model_dim, self.num_head, self.ff_dim, self.dropout_p) for i in range(self.num_layers)])
    self.dropout = torch.nn.Dropout(self.dropout_p)


  def forward(self, input, encoder_output, src_mask, time_step=0, layer_cache=None):
    tgt_mask = (input == torch.zeros(input.size(), dtype=torch.long).to(device))
    embedded = self.dropout(self.embedding(input))

    if time_step==0 and input.size(1)!=1:
      pos_id = torch.LongTensor([list(range(input.size(1))) for i in range(input.size(0))]).to(device)
    else:
      pos_id = torch.LongTensor([[time_step] for i in range(input.size(0))]).to(device)
    pos_emb = self.pos_embedding(pos_id)
    x = self.dropout(embedded + pos_emb)
    x = x.masked_fill(tgt_mask.unsqueeze(-1).expand(-1, -1, x.size(-1)), 0.0)

    for i, layer in enumerate(self.layers):
      x = layer(x, encoder_output, src_mask, tgt_mask,
                layer_cache=layer_cache[i] if layer_cache!=None else None)
    return x

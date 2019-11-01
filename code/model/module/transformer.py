import torch, math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class PositionalEncoding(torch.nn.Module):
  def __init__(self, emb_dim, dropout_p=0.1, max_len=1000): # if change max_len, loading models may be compricated
    super(PositionalEncoding, self).__init__()
    self.dropout = torch.nn.Dropout(p=dropout_p)

    pe = torch.zeros(max_len, emb_dim)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0.0, emb_dim, 2) * -(math.log(10000.0) / emb_dim))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x, mask, time_step=0):
    x = x + self.pe[:, time_step:time_step+x.size(1)]
    x.masked_fill_(mask.unsqueeze(-1).expand(-1, -1, x.size(-1)), 0.0)
    return self.dropout(x)



class MultiHeadedAttention(torch.nn.Module):
  def __init__(self, num_head, model_dim, dropout_p=0.1):
    super(MultiHeadedAttention, self).__init__()

    assert model_dim % num_head == 0
    self.model_dim = model_dim
    self.num_head = num_head
    self.dim_per_head = model_dim // num_head

    self.linear_keys = torch.nn.Linear(model_dim, num_head * self.dim_per_head)
    self.linear_values = torch.nn.Linear(model_dim, num_head * self.dim_per_head)
    self.linear_querys = torch.nn.Linear(model_dim, num_head * self.dim_per_head)
    self.final_linear = torch.nn.Linear(model_dim, model_dim)


  def forward(self, query, key, value, q_mask, kv_mask, use_subseq_mask=False, layer_cache=None):
    batch_size = query.size(0)

    if layer_cache != None:
      if torch.is_tensor(layer_cache["kv_mask"]):
        kv_mask = torch.cat((layer_cache["kv_mask"], kv_mask), 1)
        key = torch.cat((layer_cache["key"], key), 1)
        value = torch.cat((layer_cache["value"], value), 1)

      layer_cache["kv_mask"] = kv_mask
      layer_cache["key"] = key
      layer_cache["value"] = value

    q_mask_here = q_mask.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_head, -1, self.dim_per_head)
    query = self.linear_querys(query).view(batch_size, -1, self.num_head, self.dim_per_head).transpose(1, 2).masked_fill(q_mask_here, 0.0)

    kv_mask_here = kv_mask.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_head, -1, self.dim_per_head)
    key = self.linear_keys(key).view(batch_size, -1, self.num_head, self.dim_per_head).transpose(1, 2).masked_fill(kv_mask_here, 0.0)
    value = self.linear_values(value).view(batch_size, -1, self.num_head, self.dim_per_head).transpose(1, 2).masked_fill(kv_mask_here, 0.0)

    x, attn_weights = self.attention(query, key, value, kv_mask, use_subseq_mask)
    x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_head * self.dim_per_head)
    x = self.final_linear(x)

    q_mask_here = q_mask.unsqueeze(-1).expand(-1, -1, x.size(-1))
    x = x.masked_fill(q_mask_here, 0.0)

    return x, attn_weights


  def attention(self, query, key, value, mask, use_subseq_mask=False):
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.dim_per_head)

    mask = mask.unsqueeze(1).unsqueeze(1).expand(-1, scores.size(1), scores.size(2), -1)
    if use_subseq_mask:
      mask = self.get_subseq_mask(mask.size())

    attn_weights = torch.nn.functional.softmax(scores.masked_fill(mask, float('-inf')), dim = -1)
    x = torch.matmul(attn_weights, value)
    return x, attn_weights


  def get_subseq_mask(self, size):
    k = 1 + size[-1] - size[-2]
    subsequent_mask = np.triu(np.ones(size), k=k).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask).to(device)
    return subsequent_mask



class PositionwiseFeedForward(torch.nn.Module):
  def __init__(self, model_dim, ff_dim, dropout_p=0.1):
    super(PositionwiseFeedForward, self).__init__()
    self.w_1 = torch.nn.Linear(model_dim, ff_dim)
    self.w_2 = torch.nn.Linear(ff_dim, model_dim)
    self.dropout = torch.nn.Dropout(dropout_p)

  def forward(self, x):
    return self.w_2(self.dropout(torch.nn.functional.relu(self.w_1(x))))



class TransformerEncoderLayer(torch.nn.Module):
  def __init__(self, model_dim, num_head, ff_dim, dropout_p=0.1):
    super(TransformerEncoderLayer, self).__init__()
    self.self_attn = MultiHeadedAttention(num_head, model_dim, dropout_p)
    self.layer_norm1 = torch.nn.LayerNorm(model_dim, eps=1e-6)
    self.feed_forward = PositionwiseFeedForward(model_dim, ff_dim, dropout_p)
    self.layer_norm2 = torch.nn.LayerNorm(model_dim, eps=1e-6)
    self.dropout = torch.nn.Dropout(dropout_p)

  def forward(self, input, mask):
    attn_out, _ = self.self_attn(input, input, input, mask, mask)
    out = self.layer_norm1(self.dropout(attn_out) + input)
    ff_out = self.feed_forward(out).masked_fill(mask.unsqueeze(-1).expand(-1, -1, out.size(-1)), 0.0)
    out = self.layer_norm2(self.dropout(ff_out) + out)
    return out



class TransformerEncoder(torch.nn.Module):
  def __init__(self, vocab_size, emb_dim, model_dim, ff_dim, num_layers, num_head, dropout_p=0.1, padding_idx=0):
    super(TransformerEncoder, self).__init__()

    self.vocab_size = vocab_size
    self.emb_dim = emb_dim
    self.model_dim =  model_dim
    self.ff_dim = ff_dim
    self.num_layers = num_layers
    self.num_head = num_head

    self.dropout_p = dropout_p

    self.embedding = torch.nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=padding_idx)
    self.positinal_enc = PositionalEncoding(self.emb_dim, self.dropout_p)
    self.layers = torch.nn.ModuleList([TransformerEncoderLayer(self.model_dim, self.num_head, self.ff_dim, self.dropout_p) for i in range(self.num_layers)])
    self.dropout = torch.nn.Dropout(self.dropout_p)


  def forward(self, input):
    mask = (input==torch.zeros(input.size(), dtype=torch.long).to(device))
    embedded = self.dropout(self.embedding(input))
    x = self.positinal_enc(embedded, mask)
    for layer in self.layers:
      x = layer(x, mask)
    return x



class TransformerDecoderLayer(torch.nn.Module):
  def __init__(self, model_dim, num_head, ff_dim, dropout_p=0.1):
    super(TransformerDecoderLayer, self).__init__()

    self.self_attn = MultiHeadedAttention(num_head, model_dim, dropout_p)
    self.layer_norm1 = torch.nn.LayerNorm(model_dim, eps=1e-6)
    self.context_attn = MultiHeadedAttention(num_head, model_dim, dropout_p)
    self.layer_norm2 = torch.nn.LayerNorm(model_dim, eps=1e-6)
    self.feed_forward = PositionwiseFeedForward(model_dim, ff_dim, dropout_p)
    self.layer_norm3 = torch.nn.LayerNorm(model_dim, eps=1e-6)
    self.dropout = torch.nn.Dropout(dropout_p)

  def forward(self, input, encoder_output, src_mask, tgt_mask, layer_cache=None):
    attn_out, _ = self.self_attn(input, input, input, tgt_mask, tgt_mask, use_subseq_mask=True, layer_cache=layer_cache)
    out = self.layer_norm1(self.dropout(attn_out) + input)

    context_out, _ = self.context_attn(out, encoder_output, encoder_output, tgt_mask, src_mask)
    out = self.layer_norm2(self.dropout(context_out) + out)

    ff_out = self.feed_forward(out).masked_fill(tgt_mask.unsqueeze(-1).expand(-1, -1, out.size(-1)), 0.0)
    out = self.layer_norm3(self.dropout(ff_out) + out)
    return out



class TransformerDecoder(torch.nn.Module):
  def __init__(self, vocab_size, emb_dim, model_dim, ff_dim, num_layers, num_head, dropout_p=0.1, padding_idx=0):
    super(TransformerDecoder, self).__init__()

    self.vocab_size = vocab_size
    self.emb_dim = emb_dim
    self.model_dim =  model_dim
    self.ff_dim = ff_dim
    self.num_layers = num_layers
    self.num_head = num_head

    self.dropout_p = dropout_p

    self.embedding = torch.nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=padding_idx)
    self.positinal_enc = PositionalEncoding(self.emb_dim, self.dropout_p)
    self.layers = torch.nn.ModuleList([TransformerDecoderLayer(self.model_dim, self.num_head, self.ff_dim, self.dropout_p) for i in range(self.num_layers)])
    self.dropout = torch.nn.Dropout(self.dropout_p)


  def forward(self, input, encoder_output, src_mask, time_step=0, layer_cache=None):
    tgt_mask = (input == torch.zeros(input.size(), dtype=torch.long).to(device))
    embedded = self.dropout(self.embedding(input))
    x = self.positinal_enc(embedded, tgt_mask, time_step=time_step)
    for i, layer in enumerate(self.layers):
      x = layer(x, encoder_output, src_mask, tgt_mask,
                layer_cache=layer_cache[i] if layer_cache!=None else None)
    return x



class TransformerGenerator(torch.nn.Module):
  def __init__(self, vocab_size, model_dim):
    super(TransformerGenerator, self).__init__()
    self.vocab_size = vocab_size
    self.model_dim = model_dim
    self.out = torch.nn.Linear(self.model_dim, self.vocab_size)

  def forward(self, input):
    output = self.out(input)
    output = torch.nn.functional.log_softmax(output, dim=-1)
    return output

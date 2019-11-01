import torch, math
import numpy as np

from code.model.module import transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class MultiHeadedAttentionWithRelativePosition(torch.nn.Module):
  def __init__(self, num_head, model_dim, self_attn=False, k=16):
    super(MultiHeadedAttentionWithRelativePosition, self).__init__()

    assert model_dim % num_head == 0
    self.model_dim = model_dim
    self.num_head = num_head
    self.dim_per_head = model_dim // num_head

    self.linear_keys = torch.nn.Linear(model_dim, num_head * self.dim_per_head)
    self.linear_values = torch.nn.Linear(model_dim, num_head * self.dim_per_head)
    self.linear_querys = torch.nn.Linear(model_dim, num_head * self.dim_per_head)
    self.final_linear = torch.nn.Linear(model_dim, model_dim)

    self.self_attn = self_attn
    if self.self_attn:
      self.k = k
      self.wv = torch.nn.Embedding(2*self.k+1, self.dim_per_head)
      self.wk = torch.nn.Embedding(2*self.k+1, self.dim_per_head)

  def forward(self, query, key, value, q_mask, kv_mask, use_subseq_mask=False, time_step=0, layer_cache=None):
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

    x, attn_weights = self.attention(query, key, value, kv_mask, use_subseq_mask, time_step)
    x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_head * self.dim_per_head) # unite heads
    x = self.final_linear(x)

    q_mask_here = q_mask.unsqueeze(-1).expand(-1, -1, x.size(-1))
    x = x.masked_fill(q_mask_here, 0.0)

    return x, attn_weights


  def attention(self, query, key, value, mask, use_subseq_mask=False, time_step=0):
    if self.self_attn:
      length = key.size(2)
      pos = torch.LongTensor([list(range(-i, length-i)) for i in range(length)]).to(device)
      k_mat = torch.ones(pos.size(), dtype=torch.long).to(device) * self.k
      pos = torch.max(-k_mat, torch.min(k_mat, pos)) + self.k
      pos = pos[time_step:time_step+query.size(2)]

    if self.self_attn:
      ak = self.wk(pos) # [q, v, dim]

      #rel_score = query.unsqueeze(-2).expand(-1,-1,-1,length,-1) * ak.unsqueeze(0).unsqueeze(0).expand(query.size(0), query.size(1), -1, -1, -1) # [b, h, q, v, dim]
      #rel_score = torch.sum(rel_score, dim=-1) # [b, h, q, v]
      rel_score = torch.einsum('bhqd,qvd->bhqv', (query, ak))

      scores = (torch.matmul(query, key.transpose(-2, -1)) + rel_score) / math.sqrt(self.dim_per_head) # [b, h, q, v]
    else:
      scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.dim_per_head)

    mask = mask.unsqueeze(1).unsqueeze(1).expand(-1, scores.size(1), scores.size(2), -1)
    if use_subseq_mask:
      mask = self.get_subseq_mask(mask.size())

    attn_weights = torch.nn.functional.softmax(scores.masked_fill(mask, float('-inf')), dim = -1) # [b, h, q, v]

    if self.self_attn:
      av = self.wv(pos) # [q, v, dim]

      #rel_score = attn_weights.unsqueeze(-1).expand(-1,-1,-1,length,-1) * av.unsqueeze(0).unsqueeze(0).expand(attn_weights.size(0), attn_weights.size(1), -1, -1, -1) # [b, h, q, v, dim]
      #rel_score = torch.sum(rel_score, dim=-2) # [b, h, q, dim] ; part of context vector
      rel_score = torch.einsum('bhqv,qvd->bhqd', (attn_weights, av))

      x = torch.matmul(attn_weights, value) + rel_score # [b, h, q, dim]
    else:
      x = torch.matmul(attn_weights, value)

    return x, attn_weights


  def get_subseq_mask(self, size):
    k = 1 + size[-1] - size[-2]
    subsequent_mask = np.triu(np.ones(size), k=k).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask).to(device)
    return subsequent_mask




class RelTransformerEncoderLayer(torch.nn.Module):
  def __init__(self, model_dim, num_head, ff_dim, dropout_p=0.1):
    super(RelTransformerEncoderLayer, self).__init__()
    self.self_attn = MultiHeadedAttentionWithRelativePosition(num_head, model_dim, self_attn=True)
    self.layer_norm1 = torch.nn.LayerNorm(model_dim, eps=1e-6)
    self.feed_forward = transformer.PositionwiseFeedForward(model_dim, ff_dim, dropout_p)
    self.layer_norm2 = torch.nn.LayerNorm(model_dim, eps=1e-6)
    self.dropout = torch.nn.Dropout(dropout_p)

  def forward(self, input, mask):
    attn_out, _ = self.self_attn(input, input, input, mask, mask)
    out = self.layer_norm1(self.dropout(attn_out) + input)
    ff_out = self.feed_forward(out).masked_fill(mask.unsqueeze(-1).expand(-1, -1, out.size(-1)), 0.0)
    out = self.layer_norm2(self.dropout(ff_out) + out)
    return out



class RelTransformerEncoder(torch.nn.Module):
  def __init__(self, vocab_size, emb_dim, model_dim, ff_dim, num_layers, num_head, dropout_p=0.1, padding_idx=0):
    super(RelTransformerEncoder, self).__init__()

    self.vocab_size = vocab_size
    self.emb_dim = emb_dim
    self.model_dim =  model_dim
    self.ff_dim = ff_dim
    self.num_layers = num_layers
    self.num_head = num_head

    self.dropout_p = dropout_p

    self.embedding = torch.nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=padding_idx)
    #self.positinal_enc = transformer.PositionalEncoding(self.emb_dim, self.dropout_p)
    self.layers = torch.nn.ModuleList([RelTransformerEncoderLayer(self.model_dim, self.num_head, self.ff_dim, self.dropout_p) for i in range(self.num_layers)])
    self.dropout = torch.nn.Dropout(self.dropout_p)


  def forward(self, input):
    mask = (input==torch.zeros(input.size(), dtype=torch.long).to(device))
    embedded = self.dropout(self.embedding(input)) # [B,T,H]

    #x = self.positinal_enc(embedded, mask)
    x = embedded

    for layer in self.layers:
      x = layer(x, mask)
    return x




class RelTransformerDecoderLayer(torch.nn.Module):
  def __init__(self, model_dim, num_head, ff_dim, dropout_p=0.1):
    super(RelTransformerDecoderLayer, self).__init__()

    self.self_attn = MultiHeadedAttentionWithRelativePosition(num_head, model_dim, self_attn=True)
    self.layer_norm1 = torch.nn.LayerNorm(model_dim, eps=1e-6)
    self.context_attn = MultiHeadedAttentionWithRelativePosition(num_head, model_dim, self_attn=False)
    self.layer_norm2 = torch.nn.LayerNorm(model_dim, eps=1e-6)
    self.feed_forward = transformer.PositionwiseFeedForward(model_dim, ff_dim, dropout_p)
    self.layer_norm3 = torch.nn.LayerNorm(model_dim, eps=1e-6)
    self.dropout = torch.nn.Dropout(dropout_p)

  def forward(self, input, encoder_output, src_mask, tgt_mask, time_step=0,layer_cache=None):
    attn_out, _ = self.self_attn(input, input, input, tgt_mask, tgt_mask, use_subseq_mask=True, time_step=time_step, layer_cache=layer_cache)
    out = self.layer_norm1(self.dropout(attn_out) + input)

    context_out, _ = self.context_attn(out, encoder_output, encoder_output, tgt_mask, src_mask)
    out = self.layer_norm2(self.dropout(context_out) + out)

    ff_out = self.feed_forward(out).masked_fill(tgt_mask.unsqueeze(-1).expand(-1, -1, out.size(-1)), 0.0)
    out = self.layer_norm3(self.dropout(ff_out) + out)
    return out



class RelTransformerDecoder(torch.nn.Module):
  def __init__(self, vocab_size, emb_dim, model_dim, ff_dim, num_layers, num_head, dropout_p=0.1, padding_idx=0):
    super(RelTransformerDecoder, self).__init__()

    self.vocab_size = vocab_size
    self.emb_dim = emb_dim
    self.model_dim =  model_dim
    self.ff_dim = ff_dim
    self.num_layers = num_layers
    self.num_head = num_head

    self.dropout_p = dropout_p

    self.embedding = torch.nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=padding_idx)
    #self.positinal_enc = transformer.PositionalEncoding(self.emb_dim, self.dropout_p)
    self.layers = torch.nn.ModuleList([RelTransformerDecoderLayer(self.model_dim, self.num_head, self.ff_dim, self.dropout_p) for i in range(self.num_layers)])
    self.dropout = torch.nn.Dropout(self.dropout_p)


  def forward(self, input, encoder_output, src_mask, time_step=0, layer_cache=None):
    tgt_mask = (input == torch.zeros(input.size(), dtype=torch.long).to(device))
    embedded = self.dropout(self.embedding(input))
    #x = self.positinal_enc(embedded, tgt_mask, time_step=time_step)
    x = embedded

    for i, layer in enumerate(self.layers):
      x = layer(x, encoder_output, src_mask, tgt_mask, time_step=time_step,
                layer_cache=layer_cache[i] if layer_cache!=None else None)
    return x

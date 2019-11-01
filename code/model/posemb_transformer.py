import torch

from code.model.transformer_base import TransformerBaseModel
from code.model.module import transformer
from code.model.module import posemb_transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class TransformerModel(TransformerBaseModel):
  def __init__(self, setting, src_lang, tgt_lang):
    super(TransformerModel, self).__init__()
    self.src_lang, self.tgt_lang = src_lang, tgt_lang
    self.encoder = posemb_transformer.PosEmbTransformerEncoder(
                     src_lang.vocab_size,
                     setting["encoder_vars"]["emb_dim"],
                     setting["encoder_vars"]["model_dim"],
                     setting["encoder_vars"]["ff_dim"],
                     setting["encoder_vars"]["num_layers"],
                     setting["encoder_vars"]["num_head"],
                     dropout_p=setting["train_vars"]["dropout_p"],
                     padding_idx=src_lang.vocab2index["PADDING"])

    self.decoder = posemb_transformer.PosEmbTransformerDecoder(
                     tgt_lang.vocab_size,
                     setting["decoder_vars"]["emb_dim"],
                     setting["decoder_vars"]["model_dim"],
                     setting["decoder_vars"]["ff_dim"],
                     setting["decoder_vars"]["num_layers"],
                     setting["decoder_vars"]["num_head"],
                     dropout_p=setting["train_vars"]["dropout_p"],
                     padding_idx=src_lang.vocab2index["PADDING"])

    self.generator = transformer.TransformerGenerator(
                       tgt_lang.vocab_size,
                       setting["decoder_vars"]["model_dim"])

    if setting["options"]["share_embedding"]:
      if src_lang.path == tgt_lang.path:
        self.decoder.embedding = self.encoder.embedding
      else:
        raise

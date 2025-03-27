import torch.nn as nn
from torch.nn.functional import log_softmax


class EncoderDecoder(nn.Module):
    """Base class of Encoder - Decoder architecture    
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """Init the Encoder-Decoder architecture model
        Args:
            encoder (nn.Module): the encoder model
            decoder (nn/Module): the decoder model
            src_embed (_type_): embeddings of source sequences
            tgt_embed (_type_): embeddings of target sequences
            generator (_type_): the generative cap on the top of the decoder
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.scr_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and tgt sequences

        Args:
            src (_type_): _description_
            tgt (_type_): _description_
            src_mask (_type_): _description_
            tgt_mask (_type_): _description_
        """
        # Encode
        memory = self.encode(src, src_mask)
        # Decode
        return self.decode(memory, src_mask, tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decode(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
    

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # this is the projection layer to map d_model dim to the dim of vocabulary size
        self.proj = nn.Linear(d_model, vocab)
    
    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)
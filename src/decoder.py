import torch.nn as nn
from helpers import clones
from layer_norm import LayerNorm, SublayerConnection


class Decoder(nn.Module):
    """N layer of decoder_layer with masking
    """
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
    
    
class DecoderLayer(nn.Module):
    """A decoder layer: self-attention, memory-attention, feed forward
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.residual1 = SublayerConnection(size, dropout)
        self.residual2 = SublayerConnection(size, dropout)
        self.residual3 = SublayerConnection(size, dropout)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        # lower part of the decoder figure: self-attention with subsequent mask 
        x = self.residual1(x, lambda x: self.self_attn(x, x, x, tgt_mask)) 
        # middle part of the decoder figure" source-attention using memory from the encoder (Q, K, V as input)
        x = self.residual2(x, lambda x: self.src_attn(x, memory, memory, src_mask))
        # upper part of the decoder figure
        x = self.residual3(x, self.feed_forward)
        return x
        
    
import torch
import torch.nn as nn
import copy
from layer_norm import LayerNorm, SublayerConnection


class Encoder(nn.Module):
    """The encoder is a stack of N layers

    Args:
        nn (_type_): _description_
    """
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = Encoder._clone(layer, N)
        self.norm = LayerNorm(layer.size)
        

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)       

        
    @staticmethod
    def _clone(module, N):
        """Repeat module N times
        """
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
    

class EncoderLayer(nn.Module):
    """Encoder is made up of self-attention and feed forward
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # Sublayer1 creates residual link wrapping self-attention block
        self.sublayer1 = SublayerConnection(size, dropout)
        # Sublayer2 creates residual link wrapping feed forward block
        self.sublayer2 = SublayerConnection(size, dropout)
        self.size = size
        
    def forward(self, x, mask):
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer2(x, self.feed_forward)
    
    
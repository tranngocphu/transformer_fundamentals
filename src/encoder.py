import torch
import torch.nn as nn
import copy


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
    
    

class LayerNorm(nn.Module):
    """A LayerNorm module"""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
    
class SublayerConnection(nn.Module):
    """This implement the output of each Sublayer as follows.
            output = LayerNorm( x + Sublayer(x) ),
        which also has residual connection. `Sublayer` is a 
        torch module itself to be provided during forward pass.
        
        - A residual connection followed by a layer norm.
        - Note for code simplicity the norm is first as opposed to last.
        - Sublayer in this context will be a EncoderLayer (i.e., multi-head self-attention layer)
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()        
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        # Alternative:
        # return self.norm(x + self.dropout(sublayer(x)))        
        return x + self.dropout(sublayer(self.norm(x)))
    
    

class EncoderLayer(nn.module):
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
    
    
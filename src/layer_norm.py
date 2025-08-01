import torch
import torch.nn as nn


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
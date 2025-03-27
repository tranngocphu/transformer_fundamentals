import torch
import torch.nn as nn


def attention(query, key, value, mask=None, dropout=None):
    """Compute scaled dot product attention scores
        - query, key and value have shape of [batch_size, seq_len, d_k]        
    """
    d_k = query.size(dim=-1)
    
    # calculate scaled attention scores below:
    #  - resultant shape is [batch_size, seq_len, seq_len]
    #  - where dim 1 represent query, and dim -1 represent key
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(d_k) 
    
    # masking
    if mask is not None:
        # wherever mask is zero, replace elements in scores with very negative numbers
        scores = scores.masked_fill(mask==0, -1e9)
    
    # Ok, now why taking softmax along last dim below?
    # For each query token, we want probability distribution over all key tokens
    # telling us how much attention that query token should pay to each key token.
    # This distribution must sum to 1 across the key token (last dim).
    # In specific: sum(scores[b, i, :]) = 1, for specific positions b and i.
    p_attn = scores.softmax(dim=-1)
    
    if dropout is not None:
        p_attn = dropout(p_attn)

    # get the result        
    result = torch.matmul(p_attn, value)
        
    return result, p_attn
    
        
    
    
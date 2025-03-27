import torch
import torch.nn as nn
import math


def attention(query, key, value, mask=None, dropout=None):
    """Compute scaled dot product attention scores
        - query, key and value have shape of [batch_size, seq_len, d_k]        
    """
    d_k = query.size(dim=-1)
    
    # calculate scaled attention scores below:
    #  - resultant shape is [batch_size, seq_len, seq_len]
    #  - where dim 1 represent query, and dim -1 represent key
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
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

    # get the result, shape will be [batch_size, seq_len, d_k]
    result = torch.matmul(p_attn, value)
        
    return result, p_attn



class MultiheadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        # d_model must be divisible by h
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.dropout = nn.Dropout(dropout)
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        """_summary_

        Args:
            query (_type_): shape of [batch_size, d_model]
            key (_type_): shape of [batch_size, d_model]
            value (_type_): shape of [batch_size, d_model]
            mask (_type_, optional): _description_. Defaults to None.
        """
        if mask is not None:
            # expand one more dimention to account for multiple heads
            # as the same mask will apply to all heads
            mask = mask.unsqueeze(0)
        
        batch_size = query.size(0)     
        
        # Linear projection of Q, K, V, creating h heads afterwards
        # note that d_k * self.h = d_model
        # and we are adding a dimension to account for h heads
        # new shape: [batch_size, h, seq_len, d_k]
        query = self.linear_q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key   = self.linear_k(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        
        # Apply attention
        # x is final self-attention of shape [batch_size, h,  seq_len, d_k], 
        # self.attn is scaled attention map of shape [batch_size, h,  seq_len, seq_len]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # Concat all heads and apply linear projection
        # new shape: [batch_size, seq_len, d_model]
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        
        # clean up
        del query
        del key
        del value
        
        return self.linear_out(x)
    
    

if __name__ == "__main__":
    from inputs import batch_size, num_heads, seq_len, d_model
    mh_attn = MultiheadAttention(num_heads, d_model)
    x = torch.randn((batch_size, seq_len, d_model))    
    y = mh_attn(x, x, x)  
    assert list(y.shape) == [batch_size, seq_len, d_model], \
        f"Output shape {y.shape} mismatched! Expected to be of {[batch_size, seq_len, d_model]}"
    print(y.shape)
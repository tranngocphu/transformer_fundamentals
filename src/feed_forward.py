import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.linear_2(x)
    

if __name__ == "__main__"  :
    from inputs import batch_size, num_heads, seq_len, d_model, d_ff
    import torch
    ff_net = PositionwiseFeedForward(d_model, d_ff)
    x = torch.randn((batch_size, seq_len, d_model))
    y = ff_net(x)
    assert list(y.shape) == [batch_size, seq_len, d_model], f"Output size {list(y.shape)} mismatched! Expected to be {[batch_size, seq_len, d_model]}."
    print(y.shape)
    

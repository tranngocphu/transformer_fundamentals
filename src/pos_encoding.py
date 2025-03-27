import torch.nn as nn
import torch
import math



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) # shape [max_len, 1]
        div_term = torch.exp( torch.arange(0, d_model, 2) * - (math.log(1000.0) / d_model) )  # shape [d_model//2]
        pe[:, 0::2] = torch.sin(position * div_term) # assign sin() to even embedding indices
        pe[:, 1::2] = torch.cos(position * div_term) # assign cos() to odd embedding incides
        pe = pe.unsqueeze(0)  # shape [1, max_len, d_model]
        self.register_buffer("pe", pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return x
    
    
    
if __name__ == "__main__":
    from inputs import d_model, x, batch_size, seq_len
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    pe_encoder = PositionalEncoding(d_model)
    x_p = pe_encoder(x)
    assert list(x_p.shape) == [batch_size, seq_len, d_model], \
        f"Shape {x_p.shape} mismatched! Expecting {[batch_size, seq_len, d_model]}."
    print(pe_encoder.pe.shape)
    
    # plot the positional encoding matrix
    fig, ax = plt.subplots(figsize=(12, 7))
    pe = pe_encoder.pe[0].numpy().T
    sns.heatmap(pe[:,:], ax=ax)
    ax.set_xlabel('Position')
    ax.set_ylabel('Embedding')
    ax.set_title('Positional Encoding for d_model=512 and max_length=5000')
    plt.tight_layout()
    fig.savefig('figures/pe.png', dpi=300)
    
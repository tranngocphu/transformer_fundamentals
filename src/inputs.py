import torch 

batch_size = 4
seq_len = 64
d_model = 512
num_heads = 8
d_ff = 2048
src_vocab = 10
tgt_vocab = 10

x = torch.randn((batch_size, seq_len, d_model))
import torch


def subsequent_mask(seq_len):
    """Mask out subsequent positions
    """
    attn_shape = (1, seq_len, seq_len)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


if __name__ == "__main__":
    from inputs import seq_len
    mask = subsequent_mask(seq_len)
    print(mask.shape)
    print(mask)
    
    
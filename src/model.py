import copy
import torch.nn as nn
from attention import MultiheadAttention
from layer_norm import LayerNorm
from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer
from encoder_decoder import EncoderDecoder, Generator
from sequence_embedding import Embeddings
from pos_encoding import PositionalEncoding
from feed_forward import PositionwiseFeedForward


c = copy.deepcopy


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    attn = MultiheadAttention(h, d_model)    
    ff = PositionwiseFeedForward(d_model, d_ff)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        decoder=Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        src_embed=nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        tgt_embed=nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        generator=Generator(d_model, tgt_vocab)
    )
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


if __name__ == "__main__":
    from inputs import src_vocab, tgt_vocab
    model = make_model(src_vocab, tgt_vocab)
    print(model)   

    # EncoderDecoder(
    #   (encoder): Encoder(
    #     (layers): ModuleList(
    #       (0-5): 6 x EncoderLayer(
    #         (self_attn): MultiheadAttention(
    #           (dropout): Dropout(p=0.1, inplace=False)
    #           (linear_q): Linear(in_features=512, out_features=512, bias=True)
    #           (linear_k): Linear(in_features=512, out_features=512, bias=True)
    #           (linear_v): Linear(in_features=512, out_features=512, bias=True)
    #           (linear_out): Linear(in_features=512, out_features=512, bias=True)
    #         )
    #         (feed_forward): PositionwiseFeedForward(
    #           (linear_1): Linear(in_features=512, out_features=2048, bias=True)
    #           (linear_2): Linear(in_features=2048, out_features=512, bias=True)
    #           (dropout): Dropout(p=0.1, inplace=False)
    #           (relu): ReLU()
    #         )
    #         (sublayer1): SublayerConnection(
    #           (norm): LayerNorm()
    #           (dropout): Dropout(p=0.1, inplace=False)
    #         )
    #         (sublayer2): SublayerConnection(
    #           (norm): LayerNorm()
    #           (dropout): Dropout(p=0.1, inplace=False)
    #         )
    #       )
    #     )
    #     (norm): LayerNorm()
    #   )
    #   (decoder): Decoder(
    #     (layers): ModuleList(
    #       (0-5): 6 x DecoderLayer(
    #         (self_attn): MultiheadAttention(
    #           (dropout): Dropout(p=0.1, inplace=False)
    #           (linear_q): Linear(in_features=512, out_features=512, bias=True)
    #           (linear_k): Linear(in_features=512, out_features=512, bias=True)
    #           (linear_v): Linear(in_features=512, out_features=512, bias=True)
    #           (linear_out): Linear(in_features=512, out_features=512, bias=True)
    #         )
    #         (src_attn): MultiheadAttention(
    #           (dropout): Dropout(p=0.1, inplace=False)
    #           (linear_q): Linear(in_features=512, out_features=512, bias=True)
    #           (linear_k): Linear(in_features=512, out_features=512, bias=True)
    #           (linear_v): Linear(in_features=512, out_features=512, bias=True)
    #           (linear_out): Linear(in_features=512, out_features=512, bias=True)
    #         )
    #         (feed_forward): PositionwiseFeedForward(
    #           (linear_1): Linear(in_features=512, out_features=2048, bias=True)
    #           (linear_2): Linear(in_features=2048, out_features=512, bias=True)
    #           (dropout): Dropout(p=0.1, inplace=False)
    #           (relu): ReLU()
    #         )
    #         (residual1): SublayerConnection(
    #           (norm): LayerNorm()
    #           (dropout): Dropout(p=0.1, inplace=False)
    #         )
    #         (residual2): SublayerConnection(
    #           (norm): LayerNorm()
    #           (dropout): Dropout(p=0.1, inplace=False)
    #         )
    #         (residual3): SublayerConnection(
    #           (norm): LayerNorm()
    #           (dropout): Dropout(p=0.1, inplace=False)
    #         )
    #       )
    #     )
    #     (norm): LayerNorm()
    #   )
    #   (scr_embed): Sequential(
    #     (0): Embeddings(
    #       (lut): Embedding(512, 10)
    #     )
    #     (1): PositionalEncoding(
    #       (dropout): Dropout(p=0.1, inplace=False)
    #     )
    #   )
    #   (tgt_embed): Sequential(
    #     (0): Embeddings(
    #       (lut): Embedding(512, 10)
    #     )
    #     (1): PositionalEncoding(
    #       (dropout): Dropout(p=0.1, inplace=False)
    #     )
    #   )
    #   (generator): Generator(
    #     (proj): Linear(in_features=512, out_features=10, bias=True)
    #   )
    # )
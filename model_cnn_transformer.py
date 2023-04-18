import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))  # 注意力
        output = torch.matmul(attn, v)  # 注意力分数乘以v

        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)
        # print('q.shape=', q.shape)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))

        q += residual
        q = self.layer_norm(q)
        return q, attn

class PositionwiseFeedForward(nn.Module): 
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):  # d_in=d_model=310, d_inner=d_ff= args.ffn_hidden=512
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.w_2(self.gelu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x

class EncoderLayer(nn.Module):   #
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, d_word_vec, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, dropout=0.1, n_position=200):

        super().__init__()

        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        # self.position_enc = d_word_vec   # 无位置编码

        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList(
            [EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])
        # '--n_layers', default=3,    type=int,   help='the number of heads in the multi-head attention network'
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # self.conv1 = nn.Conv2d(1, 1, kernel_size=1, stride=1)

    def forward(self, src_seq, src_mask, return_attns=False):
        enc_slf_attn_list = []
        # -- Forward
        # enc_output = self.dropout(self.position_enc(src_seq))
        enc_output=src_seq  # 无位置编码
        # enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class CorruptionLayer(nn.Module):
    def __init__(self, device, corrupt_probability=0.1):
        super(CorruptionLayer, self).__init__()
        self.corrupt_p = corrupt_probability
        self.device = device

    def forward(self, feature):
        bitmask = torch.cuda.FloatTensor(feature.shape).uniform_() > self.corrupt_p
        return torch.mul(feature, bitmask)


class TransformerEncoder(nn.Module): 
    """TransformerEncoder is a stack of N encoder layers.

    Args:
        sentence_len(int)   : number of samples for each input
        seq_len    (int)    : length of each sequence
        n_layers   (int)    : number of sub-encoder-layers in the encoder
        n_heads    (int)    : number of heads in the multiheadattention models
        p_drop     (float)  : dropout value
        d_ff       (int)    : dimension of the feedforward network model

    Examples:
    # >>> encoder = TransformerEncoder(vocab_size=1000, seq_len=512)
    # >>> inp = torch.arange(512).repeat(2, 1)
    # >>> encoder(inp)
    """

    def __init__(self, sentence_len, d_feature, n_layers=6, n_heads=8, p_drop=0.5, d_ff=2048):
        super(TransformerEncoder, self).__init__()
        d_k = d_v = d_feature // n_heads
        self.encoder = Encoder(n_position=sentence_len,
                               d_word_vec=d_feature, d_model=d_feature, d_inner=d_ff,
                               n_layers=n_layers, n_head=n_heads, d_k=d_k, d_v=d_v,
                               dropout=p_drop)

        self.linear = nn.Linear(d_feature, 3)  # SEED
        # self.linear = nn.Linear(d_feature, 4)  #SEED-IV
        self.softmax = nn.Softmax(dim=-1)

        self.ch_squeeze_spatial_attn = nn.Conv3d(1, 1, kernel_size=(5, 1, 1), stride=(5, 1, 1), padding=0)
        self.sig = nn.Sigmoid()

        self.spa_squeeze_channel_attn = nn.Conv3d(1, 1, kernel_size=(1, 9, 9), stride=(1, 9, 9), padding=0)
        self.linear1 = nn.Conv3d(1, 5, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0)
        self.linear2 = nn.Conv3d(5, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0)

        self.gelu = nn.GELU()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(1, 3, 3), stride=(1, 3, 3), padding=0)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(5, 1, 1), stride=(5, 1, 1), padding=0)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1, 3, 3), padding=0)
        self.relu = nn.ReLU()

        self.Linear1 = nn.Linear(128, 128)


    def forward(self, src_seq):
        src_mask = None
        Src_seq = None
        Attention = None

        for batch in range(src_seq.shape[0]):
            tt = torch.unsqueeze(src_seq[batch], 1)  # tt torch.Size([2, 1, 5, 9, 9])

            # Attn1是空间注意力
            att1 = self.ch_squeeze_spatial_attn(tt)
            Att1 = self.sig(att1)
            outt1 = tt * Att1

            # Attn2是频段注意力
            aa = nn.functional.adaptive_avg_pool2d(tt, (1, 1))
            aa1 = self.linear1(aa)
            aa1 = self.gelu(aa1)
            attn2 = self.linear2(aa1)
            Attn2 = self.sig(attn2)
            outt2 = tt * Attn2

            Out_kong_pin = outt1 + outt2

            out1 = self.relu(self.conv1(Out_kong_pin))
            out2 = self.relu(self.conv2(out1))
            out3 = self.relu(self.conv3(out2))

            out_put = self.Linear1(torch.reshape(out3, (-1,)))

            if Src_seq == None:
                Src_seq = out_put
            else:
                Src_seq = torch.vstack((Src_seq, out_put))

        if src_seq.shape[0] == 1:
            SRC_seq = torch.unsqueeze(Src_seq, 0)
            SRC_seq = torch.unsqueeze(SRC_seq, 0)
        else:
            SRC_seq = torch.unsqueeze(Src_seq, 1)

        outputs_feature, *_ = self.encoder(SRC_seq, src_mask)
        outputs, _ = torch.max(outputs_feature, dim=1)
        outputs_classification = self.softmax(self.linear(outputs))
        return outputs_feature, outputs_classification, Attention

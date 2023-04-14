import mindspore as ms
import mindspore.nn as nn
from mindspore import ops

class Encoder(nn.Cell):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)#Embedding
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)#GRU
        self.fc = nn.Dense(enc_hid_dim * 2, dec_hid_dim)#全连接层
        self.dropout = nn.Dropout(dropout)#激活函数

    def construct(self, src):
        input_perm = (0, 2, 1)#用于transpose
        src = ops.transpose(src, input_perm)  # src = [batch_size, src_len]
        embedded = ops.transpose(self.dropout(self.embedding(src)), input_perm)  # embedded = [src_len, batch_size, emb_dim]
        # enc_output = [src_len, batch_size, hid_dim * num_directions]
        # enc_hidden = [n_layers * num_directions, batch_size, hid_dim]
        enc_output, enc_hidden = self.rnn(embedded)#获取encoder output和encoder hidden

        # enc_hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # enc_output are always from the last layer

        # enc_hidden [-2, :, : ] is the last of the forwards RNN
        # enc_hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        # encoder RNNs fed through a linear layer
        # s = [batch_size, dec_hid_dim]
        op = ops.Concat(1)#用于Concat
        s = ops.Tanh(self.fc(op((enc_hidden[-2, :, :], enc_hidden[-1, :, :]))))#tanh

        return enc_output, s
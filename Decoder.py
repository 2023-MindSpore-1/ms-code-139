import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
class Decoder(nn.Cell):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim#输出层维度
        self.attention = attention#attention
        self.embedding = nn.Embedding(output_dim, emb_dim)#Embedding
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)#GRU
        self.fc_out = nn.Dense((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)#全连接层
        self.dropout = nn.Dropout(dropout)#激活函数
        #self.fc=nn.Linear(1,output_dim)

    def construct(self, dec_input, s, enc_output):
        dec_input = ops.ExpandDims(dec_input,1)  # dec_input = [batch_size, 1]
        input_perm = (0, 2, 1)#用于transpose操作
        embedded = self.dropout(ops.transpose(self.embedding(dec_input)), input_perm)# embedded = [1, batch_size, emb_dim]
        a = ops.ExpandDims(self.attention(s, enc_output),1)# a = [batch_size, 1, src_len]
        enc_output = ops.transpose(enc_output, input_perm)# enc_output = [batch_size, src_len, enc_hid_dim * 2]  
        c = ops.transpose(ops.BatchMatMul(a, enc_output), input_perm)# c = [1, batch_size, enc_hid_dim * 2]      
        op = ops.Concat(2)#用于Concat操作
        rnn_input = op((embedded, c))# rnn_input = [1, batch_size, (enc_hid_dim * 2) + emb_dim]   
        dec_output, dec_hidden = self.rnn(rnn_input, ops.ExpandDims(s,0))#获取decoder output和hidden层  
        embedded = ops.Squeeze(embedded,0)# embedded = [batch_size, emb_dim]
        dec_output = ops.Squeeze(dec_output,0)# dec_output = [batch_size, dec_hid_dim]
        c = ops.Squeeze(c,0)# c = [batch_size, enc_hid_dim * 2]   
        op = ops.Concat(1)
        pred = self.fc_out(op((dec_output, c, embedded)))# pred = [batch_size, output_dim]
        return pred, ops.Squeeze(dec_hidden,0)
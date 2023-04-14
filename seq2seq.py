import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
import numpy as np
import process_NYC_Toyko_data as pro
import random
class Seq2Seq(nn.Cell):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder#encoder
        self.decoder = decoder#decoder
        self.device = device#device
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]#batch size
        trg_len = trg.shape[0]# data length
        trg_vocab_size = self.decoder.output_dim #decoder output 维度
        outputs = ops.Zeros((trg_len, batch_size, trg_vocab_size),ms.float32).to(self.device)#初始化output
        enc_output, s = self.encoder(src)#encoder output 和hidden
        dec_input = trg[0, :]#decoder input
        #初始化kalman filter参数A、B、P、Q、R_k
        a = np.array([[1,1],[0,1]])
        A = ms.Tensor(a)
        b = np.array([[1, 0], [0, 1]])
        B = ms.Tensor(b)
        P = ms.Tensor(b)
        q = np.array([[0.3, 0], [0, 0.3]])
        Q = ms.Tensor(q)
        R_k = ops.Zeros(2,2)
        I=np.eye(2)#单位矩阵
        for t in range(0, trg_len):
            dec_output, s = self.decoder(dec_input, s, enc_output)#decoderoutput
            outputs[t] = dec_output#赋值
            softmax=ops.Softmax(outputs[t])#softmax
            index,max=ops.ArgMaxWithValue()(softmax)#找max值
            pre_grid = [int(max.indices[i]) + 1 for i in range(0, len(max.indices))]#获取网格
            for k in range(0,len(pre_grid)):
                lat, long = pro.grid2Coordinate(pre_grid[k])#获取经纬度
                z = ms.Tensor([[lat], [long]])
                if t==0:
                    g_mat=ms.Tensor([[lat-0.01],[long-0.01]])
                #kalman filter预测和更新步骤
                g_pre=A *g_mat
                R_pre=A*R_k+P
                K_k=R_pre*B.T*(B*R_pre*B.T+Q)
                g_mat=g_pre+K_k*(z-B*g_pre)
                R_k=(I-K_k*B)*R_pre
            outputs[t]=g_mat#更新outputs
            teacher_force = random.random() < teacher_forcing_ratio#训练模式
            top1 = dec_output.argmax(1)#获取最大值
            dec_input = trg[t] if teacher_force else top1#获取decoder input

        return outputs
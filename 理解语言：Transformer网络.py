import math
import torch
from torch import Tensor, nn
from torch.nn import Transformer
from typing import Tuple

# 定义模型参数
d_model = 512  # 模型的维度
nhead = 8  # 多头注意力模型的头数
num_encoder_layers = 6  # 编码器层的数量
num_decoder_layers = 6  # 解码器层的数量

# 初始化Transformer模型
model = Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers)

# 假设我们有一个输入序列，它的形状是(10, 32)，表示有32个序列，每个序列有10个词
src = torch.rand(10, 32, d_model)

# 通过模型传递输入
out = model(src, src)

print(out.shape)  # 输出：torch.Size([10, 32, 512])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)
        self.init_weights()
    
    def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
        bptt = 5  # 每个序列的长度
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].reshape(-1)
        return data, target

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = None # 词汇表

ntokens = len(vocab) # 词汇表的大小
emsize = 200 # 嵌入维度
d_hid = 200 # 隐藏层的维度
nlayers = 2 # TransformerEncoderLayer的层数
nhead = 2 # 多头注意力机制的头数
dropout = 0.2 # 丢弃概率
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 5.0 # 学习率
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

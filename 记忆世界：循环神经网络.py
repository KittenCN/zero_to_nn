import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o= nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

rnn = SimpleRNN(input_size=10, hidden_size=20, output_size=10)

# 假设我们有一个长度为5的序列，每个元素是一个10维的向量
sequence = [torch.randn(1, 10) for _ in range(5)]

hidden = rnn.initHidden()
for i in range(5):
    output, hidden = rnn(sequence[i], hidden)
    print(output)

import torch
from torch import nn
from torchtext import data
from torchtext import datasets

# 定义字段
TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)

# 加载IMDb数据集
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 构建词汇表
TEXT.build_vocab(train_data, max_size=10000, vectors="glove.6B.100d")
LABEL.build_vocab(train_data)

# 创建迭代器
train_iterator, test_iterator = data.BucketIterator.splits(
(train_data, test_data),
batch_size=64,
device=device)

class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        return self.fc(hidden.squeeze(0))
    
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
NUM_EPOCHS = 5

# 创建模型
model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

# 开始训练
for epoch in range(NUM_EPOCHS):
    for batch in train_iterator:
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

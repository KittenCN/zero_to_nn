import torch
import torch.nn as nn
import torch.optim as optim

# 定义词汇表的大小和嵌入的维度
vocab_size = 10000
embedding_dim = 100

# 创建一个嵌入层
embedding = nn.Embedding(vocab_size, embedding_dim)

# 创建一个优化器
optimizer = optim.SGD(embedding.parameters(), lr=0.1)

# 创建一个损失函数
loss_fn = nn.MSELoss()

# 定义一个训练函数
def train(model, optimizer, loss_fn, inputs, targets):
    model.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()

# 定义一个测试函数
def test(model, inputs):
    with torch.no_grad():
        outputs = model(inputs)
    return outputs

# 定义一个语料库
corpus = [
    "我爱学习深度学习",
    "深度学习是一种强大的技术",
    "我喜欢使用PyTorch进行深度学习",
    "词嵌入是深度学习中的重要技术"
]

# 创建一个词汇表
word_to_ix = {}
for sentence in corpus:
    for word in sentence.split():
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
ix_to_word = {v: k for k, v in word_to_ix.items()}

# 创建输入数据和目标数据
inputs = []
targets = []
for sentence in corpus:
    words = sentence.split()
    for i in range(1, len(words) - 1):
        inputs.append(word_to_ix[words[i]])
        targets.append([word_to_ix[words[i - 1]], word_to_ix[words[i + 1]]])

inputs = torch.tensor(inputs, dtype=torch.long)
targets = torch.tensor(targets, dtype=torch.long)

# 训练模型
for epoch in range(100):
    loss = train(embedding, optimizer, loss_fn, inputs, targets)
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss}")

# 测试模型
word = "深度学习"
ix = word_to_ix[word]
embedding_vector = test(embedding, torch.tensor([ix], dtype=torch.long))
print(f"Embedding vector for '{word}': {embedding_vector}")

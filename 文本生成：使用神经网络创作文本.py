import torch
import torch.nn as nn
import torch.optim as optim

text = "Hello, welcome to the world of AI and deep learning. Let's explore the magic of neural networks together!"

from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer('basic_english')
tokens = tokenizer(text)

vocab = list(set(tokens))
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}

numericalized_text = [word_to_idx[word] for word in tokens]

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_epochs = 1000
hidden_size = 100
lr = 0.005

rnn = RNN(len(vocab), hidden_size, len(vocab))
optimizer = optim.SGD(rnn.parameters(), lr=lr)
criterion = nn.NLLLoss()

for epoch in range(n_epochs):
    hidden = rnn.initHidden()
    optimizer.zero_grad()
    loss = 0

    for i in range(len(numericalized_text)-1):
        input = torch.tensor([numericalized_text[i]], dtype=torch.float32)
        target = torch.tensor([numericalized_text[i+1]], dtype=torch.long)

        output, hidden = rnn(input, hidden)
        loss += criterion(output, target)

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

def generate_text(start_word, num_words):
    with torch.no_grad():
        input = torch.tensor([word_to_idx[start_word]], dtype=torch.float32)
        hidden = rnn.initHidden()
        output_words = [start_word]

        for i in range(num_words):
            output, hidden = rnn(input, hidden)
            _, top_idx = output.topk(1)
            input = top_idx.squeeze().detach().type(torch.float32)
            output_word = idx_to_word[top_idx.item()]
            output_words.append(output_word)

        return ' '.join(output_words)

print(generate_text('Hello', 20))

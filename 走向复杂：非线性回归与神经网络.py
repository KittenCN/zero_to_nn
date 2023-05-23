import torch
from torch import nn
from torch import optim

# 定义神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建一个神经网络实例
net = Net()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

X_train = torch.tensor([[70, 2],[100, 3],[120, 4]], dtype=torch.float32)
Y_train = torch.tensor([[30],[50],[60]], dtype=torch.float32)

# 训练神经网络
for epoch in range(100):  # loop over the dataset multiple times
    running_loss = 0.0

    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = X_train, Y_train

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    print('Epoch %d loss: %.3f' %(epoch + 1, running_loss))

print('Finished Training')

# 使用神经网络进行预测
test_data = torch.tensor([[80, 2]], dtype=torch.float32)
predictions = net(test_data)
print(predictions)

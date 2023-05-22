# 首先，我们需要引入一些必要的库
import torch
import torch.nn as nn
import torch.optim as optim
# Feature Matrix
X_train = torch.tensor([[70, 2],[100, 3],[120, 4]], dtype=torch.float32)
# Target Vector
Y_train = torch.tensor([[30],[50],[60]], dtype=torch.float32)
# Linear regression model
model = nn.Linear(in_features=2, out_features=1)
# Mean Squared Error (MSE) loss function
loss_fn = nn.MSELoss()
# Stochastic Gradient Descent (SGD) optimizer
optimizer = optim.SGD(model.parameters(), lr=0.0001)
# Train for 1000 epochs
for epoch in range(1000):
    # Forward pass
    Y_pred = model(X_train)
    # Compute loss
    loss = loss_fn(Y_pred, Y_train)
    # Zero gradients
    optimizer.zero_grad()
    # Backward pass
    loss.backward()
    # Update weights
    optimizer.step()
# New house data
X_new = torch.tensor([[80, 2]], dtype=torch.float32)
# Predict price
Y_pred = model(X_new)
print(Y_pred)

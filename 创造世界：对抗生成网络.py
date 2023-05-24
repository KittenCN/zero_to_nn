import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Generator
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.map1(x))
        x = torch.relu(self.map2(x))
        return torch.tanh(self.map3(x))

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.map1(x))
        x = torch.relu(self.map2(x))
        return torch.sigmoid(self.map3(x))

# Hyperparameters
batch_size = 100
lr = 0.0002
epochs = 10
latent_size = 64
hidden_size = 256
image_size = 784  # 28x28
num_classes = 1  # Real or Fake

# Data loader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_data = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the models
G = Generator(latent_size, hidden_size, image_size).to(device)
D = Discriminator(image_size, hidden_size, num_classes).to(device)

# Loss and optimizers
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=lr)
g_optimizer = torch.optim.Adam(G.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    for i, (images, _) in enumerate(train_loader):
        # Create labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train the discriminator
        images = images.view(-1, 784)  # Reshape the images
        outputs = D(images.to(device))
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake
        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train the generator
        outputs = D(fake_images)
        g_loss = criterion(outputs, real_labels)

        D.zero_grad()
        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    print(f'Epoch [{epoch}/{epochs}], d_loss:{d_loss.item():.4f}, g_loss:{g_loss.item():.4f}')

# Generate a batch of fake images
z = torch.randn(batch_size, latent_size).to(device)
fake_images = G(z)

# Display the fake images
import matplotlib.pyplot as plt
import numpy as np

fake_images = fake_images.view(fake_images.size(0), 1, 28, 28).to('cpu')
fake_images = fake_images.detach().numpy()

plt.figure(figsize=(10,10))
for i in range(fake_images.shape[0]):
    plt.subplot(10, 10, i+1)
    plt.imshow(fake_images[i, 0, :, :], cmap='gray')
    plt.axis('off')
plt.show()


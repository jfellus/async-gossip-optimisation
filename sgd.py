from torchvision.datasets import MNIST
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn

test_dataset = MNIST(root='data/', train=False)
dataset = MNIST(root='data/', train=True, transform=transforms.ToTensor())

train_ds, val_ds = random_split(dataset, [50000, 10000])
train_loader = DataLoader(train_ds, 128, shuffle=True)
val_loader = DataLoader(val_ds, 128)

model = nn.Linear(28*28, 10)

optimizer = torch.optim.SGD(model.parameters(), 0.001)


def evaluate():
    loss = []
    accuracy = []
    for x, y in val_loader:
        out = model(x.reshape(-1, model.in_features)) 
        loss.append(F.cross_entropy(out, y))
        _, predictions = torch.max(out, dim=1)
        accuracy.append(torch.sum(predictions == y).item() / len(y))
    loss = sum(loss)/len(loss)
    accuracy = sum(accuracy)/len(accuracy)
    return loss, accuracy

loss, accuracy = evaluate()
print("Init - val_loss: {:.4f}, val_acc: {:.4f}".format(loss, accuracy))

for epoch in range(10):
    for x, y in train_loader:
        out = model(x.reshape(-1, model.in_features))
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    loss, accuracy = evaluate()
    print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, loss, accuracy))
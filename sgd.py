from torchvision.datasets import MNIST
import random
import numpy
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

test = DataLoader(MNIST(root='data/', train=False), 128)
train = DataLoader(MNIST(root='data/', train=True, transform=transforms.ToTensor()), 128, shuffle=True)

class Node:
    def __init__(self, dataset):
        self.model = nn.Linear(28*28, 10)
        self.optimizer = torch.optim.SGD(self.model.parameters(), 0.001)
        self.data = dataset
        self.i = 0
        self.dataset = dataset
        self.w = 1
        self.S = self.model.weight * self.w

    def step(self):
        x,y = next(self.dataset)
        out = self.model(x.reshape(-1, self.model.in_features))
        loss = F.cross_entropy(out, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.S = self.model.weight * self.w
        self.i += 1

    def evaluate(self):
        loss = []
        accuracy = []
        for x, y in test:
            out = self.model(x.reshape(-1, self.model.in_features)) 
            loss.append(F.cross_entropy(out, y))
            _, predictions = torch.max(out, dim=1)
            accuracy.append(torch.sum(predictions == y).item() / len(y))
        loss = sum(loss)/len(loss)
        accuracy = sum(accuracy)/len(accuracy)
        print("{} - val_loss: {:.4f}, val_acc: {:.4f}".format(self.i, *self.evaluate()))
        return loss, accuracy

    def send(self, to):
        to.S += alpha*self.S
        to.w += alpha*self.w
        self.S *= (1-alpha)
        self.w *= (1-alpha)

n = 10
alpha = 0.5
MU = 0.5

l,r = divmod(len(train), n)

nodes = [
    Node(data) for data in random_split(train, [l]*n + [r])
]

if n<=10000:
    graph = numpy.ones((n,n)) - numpy.eye(n)
    graph /= graph.sum(axis=0)[:,numpy.newaxis]
    graph /= graph.sum(axis=1)[numpy.newaxis, :]
    graph /= n
else: 
    graph = None


for i in range(n*n*n):
    if(random.random()>MU):
        if graph is None:
            a = random.randint(0,n-1)
            b = random.randint(0,n-1)
            if a==b: continue
        else:
            a,b = divmod(numpy.random.choice(n*n, p=graph.flatten()), n)

        nodes[a].send(nodes[b])
    else:
        nodes[random.randint(0,n-1)].step()
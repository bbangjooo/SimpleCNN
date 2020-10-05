import torch
from torch import nn,cuda,optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
import math

# Settings

download_root='./mnist_data'
batch_size=64
device= 'cuda' if cuda.is_available()  else 'cpu'

# DataSet

train_set=MNIST(download_root,train=True,transform=transforms.ToTensor(),download=True)
test_set=MNIST(download_root,train=False,transform=transforms.ToTensor())

# DataLoader

train_loader=DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(dataset=test_set,batch_size=batch_size,shuffle=True)

# Model
# 정확도를 올리기 위해
# Batch nomalization, Dropout

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.layer1= nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer2= nn.Sequential(
            nn.Conv2d(32,32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer3= nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.classifier=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64*7*7,512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512,10)
        )
    def forward(self,x):
        in_size=x.size(0)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(in_size,-1)
        x = self.classifier(x)
        return x

model=Net()

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

# Train & Test
def train(epoch):
    model.train()
    for batch_idx, (data,target) in enumerate(train_loader):
        data, target=Variable(data),Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss=criterion(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 ==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test():
    model.eval()
    loss=0
    correct=0
    for data,target in test_loader:
        data, target=Variable(data),Variable(target)
        output=model(data)
        loss+=nn.functional.cross_entropy(output,target,reduction='sum').item()
        pred=output.data.max(1,keepdim=True)[1]
        correct+=pred.eq(target.data.view_as(pred)).cpu().sum()

    loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == "__main__":
    for epoch in range(0,5):
        train(epoch)
        test()
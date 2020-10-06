from torch.autograd import Variable
from torch import optim,cuda,nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch


# Settings

download_path="./mnist_data"
batch_size=64


# Datasets

train_set=MNIST(download_path,train=True,transform=transforms.ToTensor(),download=True)
test_set=MNIST(download_path,train=False,transform=transforms.ToTensor())


# Datalodaers

train_loader=DataLoader(dataset=train_set,batch_size=64,shuffle=True)
test_loader=DataLoader(dataset=test_set,batch_size=64,shuffle=True)

# Model

class Inception(nn.Module):
    def __init__(self,in_channels):
        super(Inception,self).__init__()
        #self.branch_pool = nn.Sequential(
        #    nn.AvgPool2d(kernel_size=3,padding=1),
        #    nn.Conv2d(in_channels,24,kernel_size=1)
        #)
        self.branch_pool=nn.Conv2d(in_channels,24,kernel_size=1)

        self.branch_1x1 = nn.Conv2d(in_channels,16,kernel_size=1)

        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(in_channels,16,kernel_size=1),
            nn.Conv2d(16,24,kernel_size=5,padding=2)
        )

        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_channels,16,kernel_size=1),
            nn.Conv2d(16,24,kernel_size=3,padding=1),
            nn.Conv2d(24,24,kernel_size=3,padding=1)
        )
    def forward(self,x):
        branch_pool=nn.functional.avg_pool2d(x,kernel_size=3,stride=1,padding=1)
        branch_pool=self.branch_pool(branch_pool)
        branch_1x1=self.branch_1x1(x)
        branch_5x5=self.branch_5x5(x)
        branch_3x3=self.branch_3x3(x)
        output=[branch_pool,branch_1x1,branch_5x5,branch_3x3]
        return torch.cat(output,1)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(1,10,kernel_size=5),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            Inception(in_channels=10)
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(88,20,kernel_size=5),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            Inception(in_channels=20)
        )
        self.classifier=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(8800,720),
            nn.BatchNorm1d(720),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(720,360),
            nn.BatchNorm1d(360),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  
            nn.Linear(360,180),
            nn.BatchNorm1d(180),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(180,10)
        )
    def forward(self,x):
        in_size=x.size(0)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(in_size,-1)
        x = self.classifier(x)
        return x

model=Net()
# Loss & Optimizer
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

# Train & Test

def train(epoch):
    model.train()
    for batch_idx, (data,target) in enumerate(train_loader):
        data,target=Variable(data),Variable(target)
        optimizer.zero_grad()
        output=model(data)
        loss=criterion(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test():
    model.eval()
    loss=0
    correct=0
    for data,target in test_loader:
        data,target=Variable(data),Variable(target)
        optimizer.zero_grad()
        output=model(data)
        loss+=nn.functional.cross_entropy(output,target,reduction='sum').item()
        pred=output.data.max(1,keepdim=True)[1]
        correct+=pred.eq(target.data.view_as(pred)).cpu().sum()
    loss/=len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == "__main__":
    for epoch in range(5):
        train(epoch)
        test()
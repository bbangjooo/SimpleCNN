import torch
from torch import nn,cuda,optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable

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

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5) # depth: 1, output_volume_size(filter_depth): 10, patch size: 5x5
        self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        self.conv3 = nn.Conv2d(20,30,kernel_size=4)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(30,20)
        self.fc2 = nn.Linear(20,10)

    def forward(self,x):
        in_size=x.size(0)
        x=nn.functional.relu(self.mp(self.conv1(x))) # output_size: ( 28-5 + 1 ) / 2(by MaxPooling)
        x=nn.functional.relu(self.mp(self.conv2(x)))
        x=nn.functional.relu(self.conv3(x))
        x=x.view(in_size,-1)

        x=self.fc(x)
        x=self.fc2(x)
        return nn.functional.log_softmax(x)

model=Net()

# Optimizer
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

# Train & Test

def train(epoch):
    model.train()
    for batch_idx, (data,target) in enumerate(train_loader):
        data, target=Variable(data),Variable(target)
        #print ("RAW DATA")
        #print (data.shape)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 ==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test():
    model.eval()
    test_loss=0
    correct=0
    for data,target in test_loader:
        data, target=Variable(data,volatile=True),Variable(target)
        output=model(data)
        test_loss+=nn.functional.nll_loss(output,target,size_average=False).data
        pred=output.data.max(1,keepdim=True)[1]
        correct+=pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 10):
    train(epoch)
    test()
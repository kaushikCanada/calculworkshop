import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser(description='cifar10 classification models, gpu test')
parser.add_argument('--lr', default=0.001, help='')
parser.add_argument('--max_epochs', type=int, default=4, help='')
parser.add_argument('--batch_size', type=int, default=768, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')


def main():

    args = parser.parse_args()

    transform_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset_train = CIFAR10(root='~/scratch/tmp/data', train=True, download=False, transform=transform_train)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers)
    
    class Net(nn.Module):

       def __init__(self):
          super(Net, self).__init__()

          self.conv1 = nn.Conv2d(3, 6, 5)
          self.pool = nn.MaxPool2d(2, 2)
          self.conv2 = nn.Conv2d(6, 16, 5)
          self.fc1 = nn.Linear(16 * 5 * 5, 120)
          self.fc2 = nn.Linear(120, 84)
          self.fc3 = nn.Linear(84, 10)

       def forward(self, x):
          x = self.pool(F.relu(self.conv1(x)))
          x = self.pool(F.relu(self.conv2(x)))
          x = x.view(-1, 16 * 5 * 5)
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
          x = self.fc3(x)
          return x

    net = Net().cuda() # Load model on the GPU

    criterion = nn.CrossEntropyLoss().cuda() # Load the loss function on the GPU
    optimizer = optim.SGD(net.parameters(), lr=float(args.lr))

    perf = []

    total_start = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):

       start = time.time()
       
       inputs = inputs.cuda() 
       targets = targets.cuda()

       outputs = net(inputs)
       loss = criterion(outputs, targets)

       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

       batch_time = time.time() - start

       images_per_sec = args.batch_size/batch_time

       perf.append(images_per_sec)

    total_time = time.time() - total_start

    # class Net(pl.LightningModule):

    #    def __init__(self):
    #       super(Net, self).__init__()

    #       self.conv1 = nn.Conv2d(3, 6, 5)
    #       self.pool = nn.MaxPool2d(2, 2)
    #       self.conv2 = nn.Conv2d(6, 16, 5)
    #       self.fc1 = nn.Linear(16 * 5 * 5, 120)
    #       self.fc2 = nn.Linear(120, 84)
    #       self.fc3 = nn.Linear(84, 10)

    #    def forward(self, x):
    #       x = self.pool(F.relu(self.conv1(x)))
    #       x = self.pool(F.relu(self.conv2(x)))
    #       x = x.view(-1, 16 * 5 * 5)
    #       x = F.relu(self.fc1(x))
    #       x = F.relu(self.fc2(x))
    #       x = self.fc3(x)
    #       return x

    #    def training_step(self, batch, batch_idx):
    #       x, y = batch
    #       y_hat = self(x)
    #       loss = F.cross_entropy(y_hat, y)
    #       return loss

    #    def configure_optimizers(self):
    #       return torch.optim.Adam(self.parameters(), lr=float(args.lr))

    # trainer = pl.Trainer(accelerator="gpu", devices=2, num_nodes=2, sync_batchnorm=True,
    #                     use_distributed_sampler=True, max_epochs = args.max_epochs,
    #                     strategy = "ddp", enable_progress_bar=False,
    #                     )
    # net = Net()
    # trainer.fit(net,train_loader)

if __name__=='__main__':
   main()

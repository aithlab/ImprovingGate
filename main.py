#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:40:18 2019

@author: aithlab
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import time
import matplotlib.pyplot as plt
import argparse

from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
from ur_lstm import UR_LSTM
from utils import Flatten, Accuracy

SEED = 7777
torch.manual_seed(SEED)
np.random.seed(SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('PyTorch with %s' %device)

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='./result',
                    help='Save directory of result files')
parser.add_argument('--batch_size', type=int, default=64,
                    help='The number of batch size')
parser.add_argument('--dataset', type=str, default='mnist',
                    help='Dataset')
args = parser.parse_args()

save_dir = args.save_dir
if not os.path.isdir(save_dir):
  os.makedirs(save_dir)
print("Save result files in %s" %save_dir)

batch_size = args.batch_size
n_class = 10
max_epoch = 100
lr = 1e-3

hidden_size = 512
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                            Flatten()])

dataset_loader = MNIST if args.dataset == 'mnist' else CIFAR10
dataset_ori_shape = [28,28,1] if dataset_loader == MNIST else [32,32,3]
input_dim = 1 if dataset_loader == MNIST else 3

dataset_trn = dataset_loader('./', train=True, transform=transform,download=True)
dataset_test = dataset_loader('./', train=False, transform=transform)
dataloader_trn = DataLoader(dataset_trn, batch_size, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size, shuffle=True)

# Check data.
# =============================================================================
# imgs, labels = next(iter(dataloader_trn))
# batch = 0
# plt.imshow(imgs[batch].reshape(dataset_ori_shape), cmap='gray')
# print('Label:', labels[batch].numpy())
# =============================================================================

class UR_Model(nn.Module):
  def __init__(self, input_dim, hidden_size, seq_len, num_class, num_layers=1):
    super(UR_Model, self).__init__()
    self.lstm = UR_LSTM(input_dim, hidden_size, num_layers, batch_first=True)
    self.fc1 = nn.Linear(hidden_size*seq_len, 256)
    self.fc2 = nn.Linear(256, num_class)
    
  def forward(self, x):
    batch_size, _, _ = x.shape
    x, (h_n, c_n) = self.lstm(x)
    x = x.reshape(batch_size, -1)
    x = torch.relu(self.fc1(x))
    out = self.fc2(x)
    
    return out
  
class Basic_Model(nn.Module):
  def __init__(self, input_dim, hidden_size, seq_len, num_class, num_layers=1):
    super(Basic_Model, self).__init__()
    self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
    self.fc1 = nn.Linear(hidden_size*seq_len, 256)
    self.fc2 = nn.Linear(256, num_class)
    
  def forward(self, x):
    batch_size, _, _ = x.shape
    x, (h_n, c_n) = self.lstm(x)
    x = x.reshape(batch_size, -1)
    x = torch.relu(self.fc1(x))
    out = self.fc2(x)
    
    return out

seq_len = np.prod(dataset_ori_shape[:-1])
ur_model = UR_Model(input_dim, hidden_size, seq_len, n_class).to(device)
basic_model = Basic_Model(input_dim, hidden_size, seq_len, n_class).to(device)
loss = nn.CrossEntropyLoss()

optimizer_ur = optim.Adam(ur_model.parameters(), lr)
optimizer_lstm = optim.Adam(basic_model.parameters(), lr)

losses, losses_test = [],[]
epoch_acc_trn, epoch_acc_test = [], []
for epoch in range(max_epoch):
  epoch_loss, acc_trn = [],[]
  for iter_, (imgs, labels) in enumerate(dataloader_trn):
    start_t = time.time()
    out_ur = ur_model(imgs.to(device))
    out_lstm = basic_model(imgs.to(device))
    
    acc_ur = Accuracy(out_ur, labels)
    acc_lstm = Accuracy(out_lstm, labels)
    acc_trn.append([acc_ur, acc_lstm])
    
    batch_loss = []
    out_optim = zip([out_ur, out_lstm], [optimizer_ur, optimizer_lstm])
    for out, optimizer in out_optim:
      loss_ = loss(out, labels.to(device))
      
      optimizer.zero_grad()
      loss_.backward()
      optimizer.step()
      batch_loss.append(loss_.cpu().detach().numpy())
    epoch_loss.append(batch_loss)
    iter_t = time.time() - start_t
    print('\rIteration [%4d/%4d] loss(ur): %.3f, loss(lstm): %.3f, time: %.2fs/it' \
          %(iter_+1, len(dataloader_trn), batch_loss[0], batch_loss[1], iter_t), 
          end='')
  
  acc_trn = np.stack(acc_trn).mean(axis=0)
  epoch_acc_trn.append(acc_trn)
  
  epoch_loss = np.stack(epoch_loss).mean(axis=0)
  losses.append(epoch_loss)
  
  print('\rEpoch [%3d/%3d] [avg. loss] ur: %.3f, vanilla: %.3f, [acc] ur: %.2f%%, vanilla: %.2f%%' \
        %(epoch+1, max_epoch, epoch_loss[0], epoch_loss[1], acc_trn[0]*100, acc_trn[1]*100))

  print('Testing...', end='')
  epoch_loss, acc_test = [], []
  for iter_, (imgs, labels) in enumerate(dataloader_test):
    start_t = time.time()
    out_lstm = basic_model(imgs.to(device))
    out_ur = ur_model(imgs.to(device))
    
    batch_loss = []
    out_optim = zip([out_ur, out_lstm], [optimizer_ur, optimizer_lstm])
    for out, optimizer in out_optim:
      loss_ = loss(out, labels.to(device))
      batch_loss.append(loss_.cpu().detach().numpy())
    epoch_loss.append(batch_loss)
    
    acc_ur = Accuracy(out_ur, labels)
    acc_lstm = Accuracy(out_lstm, labels)
    acc_test.append([acc_ur, acc_lstm])
    
    iter_t = time.time() - start_t
  
  acc_test = np.stack(acc_test).mean(axis=0)
  epoch_acc_test.append(acc_test)
  
  epoch_loss = np.stack(epoch_loss).mean(axis=0)
  losses_test.append(epoch_loss)
  
  print('\r[Test] [avg. loss] ur: %.3f, vanilla: %.3f, [acc] ur: %.2f%%, vanilla: %.2f%%' \
        %(epoch_loss[0], epoch_loss[1], acc_test[0]*100, acc_test[1]*100))

losses = np.stack(losses)
plt.figure()
plt.title('[Trn] Loss of UR LSTM')
plt.plot(losses[:, 0])
plt.savefig(os.path.join(save_dir, 'trn_loss_ur.jpg'))

plt.figure()
plt.title('[Trn] Loss of Vanilla LSTM')
plt.plot(losses[:, 1])
plt.savefig(os.path.join(save_dir, 'trn_loss_vanilla.jpg'))

epoch_acc_trn = np.stack(epoch_acc_trn)
plt.figure()
plt.title('[Trn] Acc. of UR LSTM')
plt.plot(epoch_acc_trn[:, 0])
plt.savefig(os.path.join(save_dir, 'trn_acc_ur.jpg'))

plt.figure()
plt.title('[Trn] Acc. of Vanilla LSTM')
plt.plot(epoch_acc_trn[:, 1])
plt.savefig(os.path.join(save_dir, 'test_acc_vanilla.jpg'))

losses_test = np.stack(losses_test)
plt.figure()
plt.title('[Test] Loss of UR LSTM')
plt.plot(losses_test[:, 0])
plt.savefig(os.path.join(save_dir, 'test_loss_ur.jpg'))

plt.figure()
plt.title('[Test] Loss of Vanilla LSTM')
plt.plot(losses_test[:, 1])
plt.savefig(os.path.join(save_dir, 'test_loss_vanilla.jpg'))

epoch_acc_test = np.stack(epoch_acc_test)
plt.figure()
plt.title('[Test] Acc. of UR LSTM')
plt.plot(epoch_acc_test[:, 0])
plt.savefig(os.path.join(save_dir, 'test_acc_ur.jpg'))

plt.figure()
plt.title('[Test] Acc. of Vanilla LSTM')
plt.plot(epoch_acc_test[:, 1])
plt.savefig(os.path.join(save_dir, 'test_acc_vanilla.jpg'))




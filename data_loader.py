# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 22:54:19 2019

@author: aithlab
"""
import os
import pickle
import torch

class MNISTStroke(torch.utils.data.Dataset):
  def __init__(self, root_path='./', train=True):
    self.mode = 'train' if train else 'test'
    if not self.preprocessingcCheck(root_path):
      data = pickle.load(open(os.path.join(root_path, "processed/%s"%self.mode), "rb"))
      self.data = self.preprocessing(data, root_path)
    else:
      data_path = os.path.join(root_path, "sequenced/%s.pkl" %self.mode)
      with open(data_path, 'rb') as f:
        self.data = pickle.load(f)
  
  def __getitem__(self, idx):
    data = self.data[idx]['input']
    seq_len = len(self.data[idx]['input'])
    label = self.data[idx]['label']

    return data, seq_len, label
  
  def preprocessingcCheck(self, path):
    return os.path.isdir(os.path.join(path, "sequenced"))
  
  def preprocessing(self, data, root_path):
    new_data = dict()
    for iter_, (num, data_) in enumerate(data.items()):    
      temp = []
      pt = torch.zeros([1,2])
      for dx, dy, eos, eod in data_['input']:
        pt = pt + torch.cat([dx.view(1,1), dy.view(1,1)], dim=1)
        temp.append(pt)
        if eod:
          break
      temp = torch.cat(temp,0)
      new_data[num] = {'input':temp, 'label':data_['label']}
      print('\rData Processing...(%5d/%5d)' %(iter_,len(data)),
            end='' if iter_+1 < len(data) else '\n')
    with open(os.path.join(root_path, "sequenced", '%s.pkl'%self.mode), 'wb') as f:
      pickle.dump(new_data, f)
    return new_data      

  def __len__(self):
    return len(self.data)
    
#def collate_fn(batch):
#  return tuple(zip(*batch))

def collate_fn(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## padd
    seqs = [torch.Tensor(t[0]) for t in batch]
    seq_lens = torch.tensor([t[1] for t in batch])
    labels = [t[2] for t in batch]
    seqs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    seqs = torch.nn.utils.rnn.pack_padded_sequence(seqs, seq_lens, 
                                                   batch_first=True, 
                                                   enforce_sorted=False)
    return seqs, seq_lens, labels
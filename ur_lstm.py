# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 22:20:05 2019

@author: aithlab
"""
import numpy as np
import torch
import torch.nn as nn

#device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class UR_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, 
                 batch_first=False, dropout=0, bidirectional=False):
        super(UR_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.ih = nn.Linear(input_size, 4 * hidden_size, bias=bias).to(device)
        self.hh = nn.Linear(hidden_size, 4 * hidden_size, bias=bias).to(device)
        
    def init_hidden(self, batch_size):
      h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
      c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
      
      u = np.random.uniform(1/self.hidden_size, 1-1/self.hidden_size, self.hidden_size)
      self.forget_bias = -torch.log(1/torch.tensor(u, dtype=torch.float)-1).to(device)
      
      return h_0, c_0

    def forward(self, x, hidden=None):
        if not hidden is None:
          h_prev, c_prev = hidden
        else:
          h_prev, c_prev = self.init_hidden(x.shape[0])
        h_prev = h_prev.view(h_prev.shape[1], -1)
        c_prev = c_prev.view(c_prev.shape[1], -1)
        
        out = []
        h_t, c_t = h_prev, c_prev
        for t in range(x.shape[1]):
          linear = self.ih(x[:,t,:]) + self.hh(h_t)

          r_t, f_t, u_t, o_t = torch.split(linear, self.hidden_size, dim=1)
          r_t = torch.sigmoid(r_t - self.forget_bias)
          f_t = torch.sigmoid(f_t + self.forget_bias)
          u_t = torch.tanh(u_t)
          o_t = torch.sigmoid(o_t)
  
          g_t = torch.mul(r_t, (1-(1-f_t)**2)) + torch.mul((1-r_t), f_t**2)
  
          c_t = torch.mul(g_t, c_t) + torch.mul((1-g_t), u_t)
          h_t = torch.mul(o_t, torch.tanh(c_t))
          out.append(h_t.unsqueeze(1))
        
        out = torch.cat(out, dim=1)
        h_t = h_t.view(self.num_layers, h_t.shape[0], self.hidden_size)
        c_t = c_t.view(self.num_layers, c_t.shape[0], self.hidden_size)
        
        # Reshape for compatibility
        if self.training and self.dropout > 0.0:
          pass
        
        return out, (h_t, c_t)
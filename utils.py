# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 01:06:37 2019

@author: aithlab
"""

class Flatten():
  def __call__(self, image):
    if image.shape[0] == 1 or image.shape[0] == 3:
      image = image.permute(1,2,0)
    image = image.reshape(-1,image.shape[2])
    
    return image
  
def Accuracy(pred, target):
  return sum(pred.argmax(dim=1).cpu().detach().numpy()==target.cpu().detach().numpy())/len(target)
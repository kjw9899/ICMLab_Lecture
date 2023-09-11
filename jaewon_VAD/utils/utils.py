#import visdom
import numpy as np
import torch
import torch as nn
from torch import nn, einsum
#from einops import rearrange

import option
args=option.parse_args()

def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32) #UCF(32,2048)
    r = np.linspace(0, len(feat), length+1, dtype=np.int) #(33,)
    for i in range(length):
        if r[i]!=r[i+1]:
            new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
        else:
            new_feat[i,:] = feat[r[i],:]
    return new_feat

def pseudo_label(loss) :
    ## Abnormal = 1 / Normal = 0
    
    if args.threshold <= loss :
        pse_label = torch.tensor([1])
        
    else :
        pse_label = torch.tensor([0])
        
    return pse_label


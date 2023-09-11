from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
from dataset import *
from model import *


import option
args = option.parse_args()

if __name__ == '__main__' :
    args = option.parse_args()
    train_loader = DataLoader(Dataset(args, test_mode=False, 
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=args.workers))
    test_loader = DataLoader(Dataset(args, test_mode=True,
                                      batchsize=args.batch_size,
                                      shuffle=False,
                                      num_workers=args.workers))
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Loss Function
    reconstruction_loss = nn.MSELoss()
    discriminator_loss = nn.BCELoss()
    
    # Model
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Optimizer
    optimizer_G = optim.AdamW(generator.parameters(), args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    optimizer_D = optim.AdamW(discriminator.parameters(), args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    
    
    
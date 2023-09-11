import torch
import torch.nn as nn

import option
args=option.parse_args()

def AttentionBlock() :
    return

class Generator(nn.Module) :
    def __init__(self, input_size, hidden=256) :
        super(Generator, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, hidden),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, input_size),
        )
        
    def forward(self, x) :
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Discriminator(nn.Module) :
    def __init__(self, input_size, output_size=1) :
        super(Discriminator, self).__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            
        )
        
    def forward(self, x) :
        x = self.discriminator(x)
        out = nn.Sigmoid(x)
        return out
    
    
    
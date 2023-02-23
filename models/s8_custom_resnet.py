import torch
import torch.nn as nn
import torch.nn.functional as F

class Custom_ResNet(nn.Module):
    def __init__(self):
        super(Custom_ResNet, self).__init__()
        
        self.prep_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        self.layer1 = self.X_seq(64, 128, 3)
        self.resblock1 = self.resblock(128, 128, 3)
        
        self.layer2 = self.X_seq(128, 256, 3, 1)
        
        self.layer3 = self.X_seq(256, 512, 3)
        self.resblock2 = self.resblock(512, 512, 3)
        
        self.pool = nn.MaxPool2d(4,4)
        self.FC = nn.Linear(512, 10, bias = False)
        
    def resblock(self, in_channels, out_channels, kernel_size):
        conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        return conv
    
    def X_seq(self, in_channels, out_channels, kernel_size, padding_val = 1):
        conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding_val, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        return conv
    
    def forward(self, x):
        
        x = self.prep_layer(x) ## Input size = 32x32, output size = 32x32
        
        x = self.layer1(x) ## Input size = 32x32, output size = 16x16
        res_1 = self.resblock1(x) ## Input size = 16x16, output size = 16x16
        x = x + res_1
        
        x = self.layer2(x) ## Input size = 16x16, output size = 8x8
        
        x = self.layer3(x) ## Input size = 8x8, output size = 4x4
        res_2 = self.resblock2(x) ## Input size = 4x4, output size = 4x4
        x = x + res_2 
        
        x = self.pool(x) ## Input size = 4x4, output size = 1x1
        x = x.view(x.size(0), -1)
        x = self.FC(x)
        
        x = x.view(-1, 10)
        return F.softmax(x, dim=-1)
        
        
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, normalization_type = "batch", stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        
        if (normalization_type == "layer"):
            self.bn1 = nn.GroupNorm(1, planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        #self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        if (normalization_type == "layer"):
            self.bn2 = nn.GroupNorm(1, planes)
        else:
            self.bn2 = nn.BatchNorm2d(planes)
        #self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(1,self.expansion*planes) if normalization_type == "layer" else nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, normalization_type, num_classes = 10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        if (normalization_type == "layer"):
            self.bn1 = nn.GroupNorm(1,64)
        else:
            self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], normalization_type, stride = 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], normalization_type, stride = 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], normalization_type, stride = 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], normalization_type, stride = 2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, normalization_type, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, normalization_type, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
def ResNet18(normalization_type = "batch"):
    if (normalization_type == "layer"):
        return ResNet(BasicBlock, [2, 2, 2, 2], "layer")
    else:
        return ResNet(BasicBlock, [2, 2, 2, 2], "batch")

def ResNet34(normalization_type = "batch"):
    if (normalization_type == "layer"):
        return ResNet(BasicBlock, [3, 4, 6, 3], "layer")
    else:
        return ResNet(BasicBlock, [3, 4, 6, 3], "batch")

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        
        # CONVOLUTION BLOCK 1A
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 10, kernel_size = (5, 5), stride = 2, padding = 1, dilation = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
        ) # output_size = 14
        
        # Global Average Pooling
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size = 14) ## Global Average Pooling
        ) # output_size = 1
        
    def forward(self, y):

        y = self.convblock1(y)
        y = self.gap(y)

        y = y.view(-1, 10)
        return F.log_softmax(y, dim=-1)

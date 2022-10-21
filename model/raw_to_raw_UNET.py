import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, num_classes=10):
        super(UNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1_1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2) 
        
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False) 
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.upv4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv4_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False)
        self.conv4_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)

        self.upv5 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv5_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.conv5_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)

        self.conv6 = nn.Conv2d(64,4,1)

    def forward(self, x): 
        # print(x.shape)

        conv1 = self.relu(self.conv1_1(x)) 
        conv1 = self.relu(self.conv1_2(conv1)) 
        pool1 = self.pool1(conv1)
        # print(pool1.shape)
        
        conv2 = self.relu(self.conv2_1(pool1))
        conv2 = self.relu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)
        # print(pool2.shape)
        
        conv3 = self.relu(self.conv3_1(pool2))
        conv3 = self.relu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)
        # print(pool3.shape)
        
        up4 = self.upv4(conv3)
        up4 = torch.cat([up4, conv2], 1)
        conv4 = self.relu(self.conv4_1(up4))
        conv4 = self.relu(self.conv4_2(conv4)) 
        # print(conv4.shape)

        up5 = self.upv5(conv4)
        up5 = torch.cat([up5, conv1], 1)
        conv5 = self.relu(self.conv5_1(up5)) 
        conv5 = self.relu(self.conv5_2(conv5)) 
        # print(conv5.shape)

        conv6 = self.conv6(conv5) 
        # print(conv6.shape)  
        out = F.interpolate(conv6, (252, 252)) 
        out = torch.clamp(out, min=0., max=1.) 
        return out
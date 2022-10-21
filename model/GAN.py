import torch
import torch.nn as nn

class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = True):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme

        self.pad = nn.ReplicationPad2d(padding) 

        if norm != "None": 
            self.norm = nn.BatchNorm2d(out_channels) 
        else: 
            self.norm = norm 

        if activation != "None": 
            self.activation = nn.ReLU(inplace = True) 
        else: 
            self.activation = activation 

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation, bias = False)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class PatchDiscriminator70(nn.Module):
    def __init__(self):
        super(PatchDiscriminator70, self).__init__()
        # Down sampling

        in_channels = 3 
        out_channels = 4 
        start_channels = 64

        self.block1 = nn.Sequential(
            Conv2dLayer(in_channels + out_channels, start_channels, 1, 1, 0),
            Conv2dLayer(start_channels, start_channels, 7, 1, 3)
        )
        self.block2 = nn.Sequential(
            Conv2dLayer(start_channels , start_channels * 2, 4, 2, 1),
            Conv2dLayer(start_channels * 2, start_channels * 2, 3, 1, 1)
        )
        self.block3 = nn.Sequential(
            Conv2dLayer(start_channels * 2, start_channels * 4, 4, 2, 1),
            Conv2dLayer(start_channels * 4, start_channels * 4, 3, 1, 1)
        )
        self.block4 = nn.Sequential(
            Conv2dLayer(start_channels * 4, start_channels * 8, 4, 2, 1),
            Conv2dLayer(start_channels * 8, start_channels * 8, 3, 1, 1)
        )
        # Final output, implemention of 70 * 70 PatchGAN
        self.final1 = Conv2dLayer(start_channels * 8, start_channels * 4, 4, 1, 1)
        self.final2 = Conv2dLayer(start_channels * 4, 1, 4, 1, 1, norm = 'none', activation = 'none')

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        # img_A: input; img_B: output
        print(img_A.shape)
        print(img_B.shape) 

        x = torch.cat((img_A, img_B), 1)                        # out: batch * 7 * 256 * 256
        block1 = self.block1(x)                                 # out: batch * 64 * 256 * 256
        block2 = self.block2(block1)                            # out: batch * 128 * 128 * 128
        block3 = self.block3(block2)                            # out: batch * 256 * 64 * 64
        x = self.block4(block3)                                 # out: batch * 512 * 32 * 32
        x = self.final1(x)                                      # out: batch * 512 * 31 * 31
        x = self.final2(x)                                      # out: batch * 1 * 30 * 30
        return block1, block2, block3, x

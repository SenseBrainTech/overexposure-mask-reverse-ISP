import torch
import torch.nn as nn
import torch.nn.functional as F

''' RGB to Raw '''
class UNet(nn.Module):
    def __init__(self, num_classes=10):
        super(UNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
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
        conv1 = self.relu(self.conv1_1(x)) 
        conv1 = self.relu(self.conv1_2(conv1)) 
        pool1 = self.pool1(conv1)
        
        conv2 = self.relu(self.conv2_1(pool1))
        conv2 = self.relu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)
        
        conv3 = self.relu(self.conv3_1(pool2))
        conv3 = self.relu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)
        
        up4 = self.upv4(conv3)
        up4 = torch.cat([up4, conv2], 1)
        conv4 = self.relu(self.conv4_1(up4))
        conv4 = self.relu(self.conv4_2(conv4)) 

        up5 = self.upv5(conv4)
        up5 = torch.cat([up5, conv1], 1)
        conv5 = self.relu(self.conv5_1(up5))
        conv5 = self.relu(self.conv5_2(conv5)) 

        conv6 = self.conv6(conv5)  
        out = F.interpolate(conv6, (252, 252))
        out = torch.clamp(out, min=0., max=1.)
        return  out 

''' Raw to RGB '''
class revUNet(nn.Module):
    def __init__(self, num_classes=10):
        super(revUNet, self).__init__()
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

        self.conv6 = nn.Conv2d(64, 12, kernel_size=1, stride=1) 

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

        conv6 = self.conv6(conv5) 

        out = nn.functional.pixel_shuffle(conv6, 2) 
        out = torch.clamp(out, min=0., max=1.) 
        return out

''' RGB to RGB '''
class resUNet(nn.Module):
    def __init__(self, num_classes=10):
        super(resUNet, self).__init__() 
        self.relu = nn.ReLU()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
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

        self.conv6 = nn.Conv2d(64, 3, kernel_size=1, stride=1) 

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

        conv6 = self.conv6(conv5) 

        out = torch.clamp(conv6, min=0., max=1.) 
        return out

''' Raw to Raw '''
class rawNet(nn.Module): 
    def __init__(self, num_classes=10):
        super(rawNet, self).__init__()
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

# ''' Raw to Raw with extra layer ''' 
# class rawNet(nn.Module):
#     def __init__(self, num_classes=10):
#         super(rawNet, self).__init__()
#         self.relu = nn.ReLU()
#         self.conv1_1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
#         self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
#         self.pool1 = nn.MaxPool2d(kernel_size=2)
        
#         self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.pool2 = nn.MaxPool2d(kernel_size=2)
        
#         self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
#         self.pool3 = nn.MaxPool2d(kernel_size=2)
        
#         self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.pool4 = nn.MaxPool2d(kernel_size=2)
        
#         self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
#         self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
#         self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
#         self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
#         self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
#         self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
#         self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
#         self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
#         self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
#         self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
#         self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
#         self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
#         self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
#         self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
#         self.conv10_1 = nn.Conv2d(32, 4, 1)
    
#     def forward(self, x):
#         conv1 = self.relu(self.conv1_1(x))
#         conv1 = self.relu(self.conv1_2(conv1))
#         pool1 = self.pool1(conv1) 
#         # print(pool1.shape)
        
#         conv2 = self.relu(self.conv2_1(pool1))
#         conv2 = self.relu(self.conv2_2(conv2)) 
#         pool2 = self.pool2(conv2)
#         # print(pool2.shape)
        
#         conv3 = self.relu(self.conv3_1(pool2))
#         conv3 = self.relu(self.conv3_2(conv3)) 
#         pool3 = self.pool3(conv3) 
#         # print(pool3.shape)
        
#         conv4 = self.relu(self.conv4_1(pool3))
#         conv4 = self.relu(self.conv4_2(conv4))
#         pool4 = self.pool4(conv4) 
#         # print(pool4.shape)
        
#         conv5 = self.relu(self.conv5_1(pool4))
#         conv5 = self.relu(self.conv5_2(conv5)) 
#         # print(conv5.shape)
        
#         up6 = self.upv6(conv5)
#         up6 = F.pad(up6, (0, 1, 1, 0), "constant", 0) 
#         up6 = torch.cat([up6, conv4], 1)
#         conv6 = self.relu(self.conv6_1(up6))
#         conv6 = self.relu(self.conv6_2(conv6)) 
#         # print(conv6.shape)
        
#         up7 = self.upv7(conv6) 
#         up7 = F.pad(up7, (0, 1, 1, 0), "constant", 0) 
#         up7 = torch.cat([up7, conv3], 1)
#         conv7 = self.relu(self.conv7_1(up7))
#         conv7 = self.relu(self.conv7_2(conv7))
#         # print(conv7.shape)
        
#         up8 = self.upv8(conv7)
#         up8 = torch.cat([up8, conv2], 1)
#         conv8 = self.relu(self.conv8_1(up8))
#         conv8 = self.relu(self.conv8_2(conv8))
#         # print(conv8.shape)
        
#         up9 = self.upv9(conv8)
#         up9 = torch.cat([up9, conv1], 1)
#         conv9 = self.relu(self.conv9_1(up9))
#         conv9 = self.relu(self.conv9_2(conv9))
#         # print(conv9.shape)
        
#         conv10 = self.conv10_1(conv9)
#         # print(conv10.shape)
 
#         out = torch.clamp(conv10, min=0., max=1.) 
#         return  out 

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init

class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = torch.qr(torch.randn(*w_shape))[0]

        if not LU_decomposed:
            self.weight = nn.Parameter(torch.Tensor(w_init))
        else:
            p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
            s = torch.diag(upper)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            upper = torch.triu(upper, 1)
            l_mask = torch.tril(torch.ones(w_shape), -1)
            eye = torch.eye(*w_shape)

            self.register_buffer("p", p)
            self.register_buffer("sign_s", sign_s)
            self.lower = nn.Parameter(lower)
            self.log_s = nn.Parameter(log_s)
            self.upper = nn.Parameter(upper)
            self.l_mask = l_mask
            self.eye = eye

        self.w_shape = w_shape
        self.LU_decomposed = LU_decomposed

    def get_weight(self, input, reverse):
        b, c, h, w = input.shape

        if not self.LU_decomposed:
            dlogdet = torch.slogdet(self.weight)[1] * h * w
            if reverse:
                weight = torch.inverse(self.weight)
            else:
                weight = self.weight
        else:
            # input.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
            # self.l_mask = self.l_mask.to(input.device) 
            # self.eye = self.eye.to(input.device) 

            self.l_mask = self.l_mask.to(input.device) 
            self.eye = self.eye.to(input.device) 

            lower = self.lower * self.l_mask + self.eye

            u = self.upper * self.l_mask.transpose(0, 1).contiguous()
            u += torch.diag(self.sign_s * torch.exp(self.log_s))

            dlogdet = torch.sum(self.log_s) * h * w

            if reverse:
                u_inv = torch.inverse(u)
                l_inv = torch.inverse(lower)
                p_inv = torch.inverse(self.p)

                weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
            else:
                weight = torch.matmul(self.p, torch.matmul(lower, u))

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)

        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        initialize_weights(self.conv5, 0)
    
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1))) 
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5

def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlock(channel_in, channel_out, init)
            else:
                return DenseBlock(channel_in, channel_out)
            # return UNetBlock(channel_in, channel_out)
        else:
            return None

    return constructor


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num # 1
        self.split_len2 = channel_num - channel_split_num # 2 

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

        in_channels = 3        
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)
        
    def forward(self, x, rev=False):
        if not rev:            
            # invert1x1conv 
            x, logdet = self.flow_permutation(x, logdet=0, rev=False) 
            
            # split to 1 channel and 2 channel. 
            x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2)) 

            y1 = x1 + self.F(x2) # 1 channel 
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1) # 2 channel 
            out = torch.cat((y1, y2), 1)
        else:
            # split. 
            x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2)) 
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s)) 
            y1 = x1 - self.F(y2) 

            x = torch.cat((y1, y2), 1)            

            # inv permutation 
            out, logdet = self.flow_permutation(x, logdet=0, rev=True)

        return out

class InvISPNet(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=subnet('DBNet'), block_num=8):
        super(InvISPNet, self).__init__()
        operations = []

        current_channel = channel_in
        channel_num = channel_in
        channel_split_num = 1

        for j in range(block_num): 
            b = InvBlock(subnet_constructor, channel_num, channel_split_num) # one block is one flow step. 
            operations.append(b)
        
        self.operations = nn.ModuleList(operations)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_() 
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
    
    def forward(self, x, rev=False):
        out = x # x: [N,3,H,W] 
        # print(self.operations) 

        if not rev: 
            for op in self.operations:
                out = op.forward(out, rev)
        else:
            for op in reversed(self.operations):
                out = op.forward(out, rev)
        
        return out

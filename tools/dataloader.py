import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torch.optim import Adam

from tools.utils import *
from tools.unprocess import * 

class LoadData(Dataset):
    def __init__(self, root, rgb_files, raw_files=None, debug=False, test=None):
        self.root = root
        self.test = test
        self.rgbs = sorted(rgb_files)
        if self.test:
            self.raws = None
        else:
            self.raws = sorted(raw_files)
        self.debug = debug
        if self.debug:
            self.rgbs = self.rgbs[:100] 
            self.raws = self.raws[:100]
        
    def __len__(self):
        return len(self.rgbs)

    def __getitem__(self, idx):
        rgb = load_img (self.rgbs[idx], norm=True) 
        # rgb = oe_mask(rgb.transpose(2, 0, 1)) 
        rgb = yuv_oe_mask(rgb.transpose(2, 0, 1)) 

        rgb = torch.from_numpy(rgb) 

        # ps = 128
        # ps_temp = ps*2 + 16 
        # H = rgb.shape[0] 
        # W = rgb.shape[1]
        # r = np.random.randint(0, H - ps_temp)
        # c = np.random.randint(0, W - ps_temp)
        # if r%2!=0: r = r-1
        # if c%2!=0: c = c-1
        # rgb_patch = rgb[r:r + ps_temp, c:c + ps_temp, :] 
        # print(rgb_patch.shape)

        # rgb = torch.from_numpy(rgb_patch.transpose((2, 0, 1))) 
        
        if self.test: 
            return rgb, self.rgbs[idx] 
        else:
            raw = load_raw (self.raws[idx]) 
            raw = torch.from_numpy(raw.transpose((2, 0, 1))) 
            return rgb, raw, self.rgbs[idx]
            
            # raw_patch = raw[r:r + ps_temp, c:c + ps_temp, :] 
            # raw = torch.from_numpy(raw_patch.transpose((2, 0, 1))) 
            # return rgb, raw

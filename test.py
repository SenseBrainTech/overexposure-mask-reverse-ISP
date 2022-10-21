import torch
import time 

from glob import glob 
from tqdm import tqdm 
from tools.utils import * 
from tools.dataloader import * 
from tools.unprocess import * 

#from isp_demosaic_net.demosaic_bayer import * 
from model.rgb_to_raw_UNET import * 
from model.nested_UNET import * 

PATH = "./"  

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
print(device) 

OE_PATH = "./trained_models/final_yuv_OE_mask-220.pt" 
NOE_PATH = "./trained_models/final_yuv_OE_NOE_mask-220.pt"

model_OE = U_Net().to(device)  
model_NOE = U_Net().to(device) 

model_OE.load_state_dict(torch.load(OE_PATH), strict=False) 
model_NOE.load_state_dict(torch.load(NOE_PATH), strict=False) 

valid_rgbs = sorted(glob('test_rgb/*'))

BATCH_TEST = 1

test_dataset = LoadData(root=PATH, rgb_files=valid_rgbs, test=True)
test_loader  = DataLoader(dataset=test_dataset, batch_size=BATCH_TEST, shuffle=True, num_workers=1,
                         pin_memory=True, drop_last=False)

cnt = 1
SUBMISSION_PATH = './results/'
runtime = [] 

psnr = [] 

print(OE_PATH, NOE_PATH) 

model_OE.eval() 
model_NOE.eval() 
with torch.no_grad():
    for (rgb_batch, rgb_name) in tqdm(test_loader):
        rgb_batch = rgb_batch.to(device) 
        rgb_name  = rgb_name[0].split('/')[-1].replace('.jpg', '') 
        
        st = time.time() 
 
        recon_rgb_OE = model_OE(rgb_batch) 
        recon_rgb_NOE = model_NOE(rgb_batch) 
        
        for i in range(BATCH_TEST): 
            for j in range(3): 
                OE_mask = rgb_batch[i,-1,:,:] 
                recon_rgb_OE[i, j][OE_mask  == 0] = 0 
                recon_rgb_NOE[i, j][OE_mask == 1] = 0 

        recon_rgb = (recon_rgb_OE + recon_rgb_NOE) 

        mosaic_raw = [] 

        # mosaic recon_raw 
        for i in range(BATCH_TEST): 
            mosaic_raw.append(mosaic(recon_rgb[i])) 

        recon_raw = torch.stack(mosaic_raw).to(device)

        tt = time.time() - st 
        runtime.append(tt) 
        
        recon_raw = recon_raw[0].detach().cpu().permute(1, 2, 0).numpy() 

        ## save as np.uint16
        # assert recon_raw.shape[-1] == 4
        recon_raw = (recon_raw * 1024).astype(np.uint16)
        np.save(SUBMISSION_PATH + rgb_name + '.npy', recon_raw)  
        cnt+=1 

print (np.mean(runtime)) 

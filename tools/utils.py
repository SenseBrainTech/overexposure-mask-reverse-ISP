import cv2
import numpy as np
import rawpy
# import matplotlib.pyplot as plt
import imageio

import torch


def extract_bayer_channels(raw):
    ch_B  = raw[1::2, 1::2] 
    ch_Gb = raw[0::2, 1::2]
    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    return ch_R, ch_Gr, ch_B, ch_Gb

def load_rawpy (raw_file):
    raw = rawpy.imread(raw_file)
    raw_image = raw.raw_image
    return raw_image

def load_img (filename, debug=False, norm=True, resize=None):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if norm:   
        img = img / 255.
        img = img.astype(np.float32) 
    if debug:
        print (img.shape, img.dtype, img.min(), img.max())
        
    if resize:
        img = cv2.resize(img, (resize[0], resize[1]), interpolation = cv2.INTER_AREA)
        
    return img

def save_rgb (img, filename):
    # if np.max(img) <= 1:
    #     img = img * 255 
    
    img = img * 255 
    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
    cv2.imwrite(filename, img)
    
def load_raw_png(raw, debug=False):
  
    
    assert '.png' in raw
    raw = np.asarray(imageio.imread((raw)))
    ch_R, ch_Gr, ch_B, ch_Gb = extract_bayer_channels (raw)

    RAW_combined = np.dstack((ch_R, ch_Gr, ch_Gb, ch_B))
    RAW_norm = RAW_combined.astype(np.float32) / (4 * 255)
    RAW_norm = np.clip(RAW_norm, 0, 1)
    
    if debug:
        print (RAW_norm.shape, RAW_norm.dtype, RAW_norm.min(), RAW_norm.max())

    # raw as (h,w,1) in RGBG domain! do not use
    raw_unpack = raw.astype(np.float32) / (4 * 255)
    raw_unpack = np.expand_dims(raw_unpack, axis=-1)
    
    return RAW_norm

def load_raw(raw, max_val=2**10):
    raw = np.load (raw)/ max_val 
    return raw.astype(np.float32)


########## RAW image manipulation

def unpack_raw(im):
    """
    Unpack RAW image from (h,w,4) to (h*2 , w*2, 1)
    """
    h,w,chan = im.shape 
    H, W = h*2, w*2
    img2 = np.zeros((H,W)) 
    img2[0:H:2,0:W:2]=im[:,:,0] 
    img2[0:H:2,1:W:2]=im[:,:,1]
    img2[1:H:2,0:W:2]=im[:,:,2]
    img2[1:H:2,1:W:2]=im[:,:,3]
    img2 = np.squeeze(img2) 
    img2 = np.expand_dims(img2, axis=-1) 
    return img2

def torch_unpack_raw(im):
    """
    Unpack RAW image from (h,w,4) to (h*2 , w*2, 1)
    """
    h,w,chan = im.shape 
    H, W = h*2, w*2
    # img2 = np.zeros((H,W)) 
    img2 = torch.zeros(H, W) 
    img2[0:H:2,0:W:2]=im[:,:,0] 
    img2[0:H:2,1:W:2]=im[:,:,1]
    img2[1:H:2,0:W:2]=im[:,:,2]
    img2[1:H:2,1:W:2]=im[:,:,3]
    # img2 = np.squeeze(img2) 
    img2 = torch.squeeze(img2) 
    # img2 = np.expand_dims(img2, axis=-1) 
    img2 = torch.unsqueeze(img2, dim=-1) 
    return img2

def pack_raw(im):
    """
    Pack RAW image from (h,w,1) to (h/2 , w/2, 4)
    """
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    ## R G G B
    out = np.concatenate((im[0:H:2,0:W:2,:], 
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:],
                       im[1:H:2,1:W:2,:]), axis=2)
    return out 

def crop(raw_patch, rgb_patch, ps):
    rgb_patch = rgb_patch.permute(1,2,0)
    raw_patch = raw_patch.permute(1,2,0)  
    
    H = raw_patch.shape[0]
    W = raw_patch.shape[1]
    r = (H - ps) // 2
    c = (W - ps) // 2
    PS, R, C = ps*2, r*2, c*2
    rgb_patch = rgb_patch[R:R + PS, C:C + PS, :]
    raw_patch = raw_patch[r:r + ps, c:c + ps, :]
    
    rgb_patch = rgb_patch.permute(2,0,1) 
    raw_patch = raw_patch.permute(2,0,1) 
    return rgb_patch, raw_patch 


########## VISUALIZATION

def demosaic (raw):
    """Simple demosaicing to visualize RAW images
    Inputs:
     - raw: (h,w,4) RAW RGGB image normalized [0..1] as float32
    Returns: 
     - Simple Avg. Green Demosaiced RAW image with shape (h*2, w*2, 3)
    """
    
    assert raw.shape[-1] == 4
    shape = raw.shape
    
    red        = raw[:,:,0]
    green_red  = raw[:,:,1]
    green_blue = raw[:,:,2] 
    blue       = raw[:,:,3]
    avg_green  = (green_red + green_blue) / 2
    image      = np.stack((red, avg_green, blue), axis=-1)
    image      = cv2.resize(image, (shape[1]*2, shape[0]*2))
    return image


def mosaic_rgb(rgb):
    """Extracts RGGB Bayer planes from an RGB image."""
    rgb = rgb.permute(1, 2, 0)
    assert rgb.shape[-1] == 3
    shape = rgb.shape
    
    red        = rgb[0::2, 0::2, 0] 
    green_red  = rgb[0::2, 1::2, 1] 
    green_blue = rgb[1::2, 0::2, 1]
    blue       = rgb[1::2, 1::2, 2]
    
    image = np.stack((red, green_red, green_blue, blue), axis=-1)
    return image


def gamma_compression(image):
    """Converts from linear to gamma space."""
    return np.maximum(image, 1e-8) ** (1.0 / 2.2) 

def tonemap(image):
    """Simple S-curved global tonemap"""
    return (3*(image**2)) - (2*(image**3))

def postprocess_raw(raw):
    """Simple post-processing to visualize demosaic RAW imgaes
    Input:  (h,w,3) RAW image normalized
    Output: (h,w,3) post-processed RAW image
    """
    raw = gamma_compression(raw)
    raw = tonemap(raw)
    raw = np.clip(raw, 0, 1)
    return raw

# def plot_pair (rgb, raw, t1='RGB', t2='RGB', axis='off'):
    
#     fig = plt.figure(figsize=(12, 6), dpi=80)
#     plt.subplot(1,2,1)
#     plt.title(t1)
#     plt.axis(axis)
#     plt.imshow(rgb)

#     plt.subplot(1,2,2)
#     plt.title(t2)
#     plt.axis(axis)
#     plt.imshow(raw)
#     plt.show()

########## METRICS

def PSNR(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2) 
    if(mse == 0):  
        return np.inf
    
    max_pixel = np.max(y_true)
    # max_pixel is negative 
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse)) 
    return psnr 

def rgb2yuv(img, fwd=True):
    '''
        the function is to convert from linearized RGB to yuv, or yuv to rgb.
        [in]
            img, [m,n,3]. np.float32. rgb or yuv
            fwd: If True, rgb-->yuv. otherwise, yuv --> rgb
        [out]
            yuv: If fwd is True, it is yuv. Otherwise, it is RGB. 

    '''
    # img should be linear in the range of [0,1]
    h, w, c, = img.shape
    assert c==3, 'the input img has to be of shape [h,w,3]'

    # M = [ .299 ,   .587 ,    .114, \
    #      -.14713 ,-.28886 ,  .436, \
    #       .615 ,  -.51499 , -.10001] 

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

    M = [.299, .587, .114, \
         -.169, -.331, .499, \
         .499, -.418, -.0813]
    M = torch.tensor(M).reshape((3,3)).type(torch.float32).to(device) 

    if fwd ==False: 
        M = torch.linalg.inv(M) 

    img = torch.reshape(img, (h*w, 3)) 
    img = img.T

    yuv = torch.matmul(M, img) 
    yuv = torch.reshape(yuv.T, (h,w,3)) 

    # y = torch.clamp(yuv[0], 0, 1) 
    # u = torch.clamp(yuv[1], -0.5, 0.5) 
    # v = torch.clamp(yuv[2], -0.5, 0.5) 

    # yuv = torch.stack([y, u, v]) 

    return yuv 

def yuv_oe_mask(img): 
    ''' 
    input: RGB 
    output: RGB with binary mask (1 if overexposed, 0 if not overexposed) 
    ''' 
    # img = np.random.rand(3, 2, 2) * 1.5
    # print(img) 

    img = img.astype(np.float32) 

    y = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2] 

    # print(y.shape) 

    y[y >= 0.978] = 1 
    y[y < 0.978] = 0 

    y = np.expand_dims(y, axis=0) 

    yuv_oe_mask = np.append(img, y, axis = 0) 

    # print(yuv_oe_mask) 
    return yuv_oe_mask 


def oe_mask(img): 
    ''' 
    input: RGB 
    output: RGB with binary mask (1 if overexposed, 0 if not overexposed) 
    ''' 
    # img [3 x 504 x 504] 
    # img = np.random.rand(3, 2, 2) * 2 
    # print(img) 

    y = np.max(img, axis=0) 

    # print(y) 

    y[y >= 1 - 1e-2] = 1; y[y < 1] = 0 
    y = np.expand_dims(y, axis=0) 

    # print(y) 

    oe_mask = np.append(img, y, axis = 0) 

    return oe_mask 



import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import math
import os
import tifffile as tiff
from skimage import color

def ProPhotoRGB2XYZ(pp_rgb,reverse=False):
    if not reverse:
        M = [[0.7976749, 0.1351917, 0.0313534], \
           [0.2880402, 0.7118741, 0.0000857], \
           [0.0000000, 0.0000000, 0.8252100]]
    else:
        M = [[ 1.34594337, -0.25560752, -0.05111183],\
             [-0.54459882,  1.5081673,   0.02053511],\
             [ 0,          0,          1.21181275]]
    M = np.array(M)
    sp = pp_rgb.shape
    xyz = np.transpose(np.dot(M, np.transpose(pp_rgb.reshape((sp[0] * sp[1], sp[2])))))
    return xyz.reshape((sp[0], sp[1], 3))

def linearize_ProPhotoRGB(pp_rgb, reverse=False):
    if not reverse:
        gamma = 1.8
    else:
        gamma = 1.0/1.8
    pp_rgb = np.power(pp_rgb, gamma)
    return pp_rgb

def XYZ_chromatic_adapt(xyz, src_white='D65', dest_white='D50'):
    if src_white == 'D65' and dest_white == 'D50':
        M = [[1.0478112, 0.0228866, -0.0501270], \
           [0.0295424, 0.9904844, -0.0170491], \
           [-0.0092345, 0.0150436, 0.7521316]]
    elif src_white == 'D50' and dest_white == 'D65':
        M = [[0.9555766, -0.0230393, 0.0631636], \
           [-0.0282895, 1.0099416, 0.0210077], \
           [0.0122982, -0.0204830, 1.3299098]]
    else:
        raise UtilCnnImageEnhanceError('invalid pair of source and destination white reference %s,%s')\
            % (src_white, dest_white)
    M = np.array(M)
    sp = xyz.shape
    assert sp[2] == 3
    xyz = np.transpose(np.dot(M, np.transpose(xyz.reshape((sp[0] * sp[1], 3)))))
    return xyz.reshape((sp[0], sp[1], 3))

def read_tiff_16bit_img_into_XYZ(tiff_fn, exposure=0):
    pp_rgb = tiff.imread(tiff_fn)
    pp_rgb = np.float64(pp_rgb) / (2 ** 16 - 1.0) 
    if not pp_rgb.shape[2] == 3:
        print('pp_rgb shape',pp_rgb.shape)
        raise UtilImageError('image channel number is not 3')
    pp_rgb = linearize_ProPhotoRGB(pp_rgb)
    pp_rgb *= np.power(2, exposure)
    xyz = ProPhotoRGB2XYZ(pp_rgb)
    xyz = XYZ_chromatic_adapt(xyz, src_white='D50', dest_white='D65')
    return xyz

def read_tiff_16bit_img_into_LAB(tiff_fn, exposure=0, normalize_Lab=False):
    xyz = read_tiff_16bit_img_into_XYZ(tiff_fn, exposure)
    lab = color.xyz2lab(xyz)
    if normalize_Lab:
        normalize_Lab_image(lab)
    return lab



def calculate_Lab_RMSE(img1, img2):
    # img1 and img2 have range [0, 255]
    #img1 = img1.astype(np.float64)#/255
    #img2 = img2.astype(np.float64)#/255
    num_pix = img1.shape[0]*img1.shape[1]
    
    Lab_RMSE = np.mean(np.sqrt(np.sum((img1 - img2)**2, axis=2)))   # correct 1
    #Lab_RMSE = np.sum(np.sqrt(np.sum((img1 - img2) ** 2, axis=2))) / num_pix   # correct 2   same with correct 1
    
    #Lab_RMSE = np.sqrt(np.sum(((img1 - img2) ** 2)) / num_pix)  # a liiter different
    
    return Lab_RMSE

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim_my(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim_my(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim_my(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim_my(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

# ##########################################################
# Please specify the paths for input dir and ground truth dir.
input_path=""
GT_path=""

input_fname_list = os.listdir(input_path)
input_fname_list.sort()
input_path_list = [os.path.join(input_path, fname) for fname in input_fname_list]

GT_fname_list = os.listdir(GT_path)
GT_fname_list.sort()
GT_path_list = [os.path.join(GT_path, fname) for fname in GT_fname_list]

assert len(input_path_list) == len(GT_path_list)
print(len(input_path_list))


psnr_list = []
ssim_list = []
Lab_RMSE_list = []
for i in range(len(input_path_list)):
    assert input_fname_list[i].split('.')[0] == GT_fname_list[i].split('.')[0]
    img1 = cv2.imread(input_path_list[i], cv2.IMREAD_COLOR)
    img2 = cv2.imread(GT_path_list[i], cv2.IMREAD_COLOR)
    
    
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    img1_lab = cv2.cvtColor(img1, cv2.COLOR_BGR2Lab)
    img2_lab = cv2.cvtColor(img2, cv2.COLOR_BGR2Lab)
    
     
    
    psnr = calculate_psnr(img1_rgb, img2_rgb)
    ssim = calculate_ssim(img1_rgb, img2_rgb)
    
    Lab_RMSE = calculate_Lab_RMSE(img1_lab, img2_lab)
    
    print('img: {}  PSNR: {}  SSIM: {}  Lab_RMSE: {}'.format(input_fname_list[i].split('.')[0], psnr, ssim, Lab_RMSE))
    
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    Lab_RMSE_list.append(Lab_RMSE)
    
print('Average PSNR: {}  SSIM: {}  Lab_RMSE: {}  Total image: {}'.format(np.mean(psnr_list), np.mean(ssim_list), np.mean(Lab_RMSE_list), len(psnr_list)))
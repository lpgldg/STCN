import argparse, time, os
import imageio
import glob
import PIL.Image as pil_image
import util
import numpy as np
import torch

from TorchTools.Functions.Metrics import PSNR, psnr, YCbCr_psnr
from modules import pytorch_ssim
#from modules.PerceptualSimilarity import perceptual_similarity


import sys
sys.path.append('../')
sys.path.append('./modules')
sys.path.append('./modules/PerceptualSimilarity')
sys.path.append('./modules/pytorch_ssim')


def main():
        scale = 2
        image1_dir = os.path.join(os.getcwd(), '/home/ilaopis/桌面/RCAN-master/RCAN_TestCode/SR/BI/RCAN/Set5/Hx2')
        image2_dir = os.path.join(os.getcwd(), '/home/ilaopis/桌面/RCAN-master/RCAN_TestCode/SR/BI/RCAN/Set5/x2')
        total_psnr = []
        total_ssim = []
        total_ssim_RGB = []
        total_lpips =[]
      
        path_list = sorted(glob.glob('{}/*'.format(image1_dir)))
        for image_path1 in path_list:
           _name =image_path1.split("/")
           _name[-2] = "x2"
           image_path2 = "/".join(_name)
           hr = pil_image.open(image_path1).convert('RGB')
           sr = pil_image.open(image_path2).convert('RGB')

           hr = np.array(hr).astype(np.float32)
           sr = np.array(sr).astype(np.float32)
           print(sr.shape)
       
            # calculate PSNR/SSIM metrics on Python
           psnr, ssim = util.calc_metrics(hr, sr, crop_border=scale)
           sr = np.transpose(sr,(1, 2, 0))
           hr = np.transpose(hr,(1, 2, 0))
           sr = torch.tensor(sr)
           hr = torch.tensor(hr)
           ssim_RGB = pytorch_ssim.ssim(sr / 255, hr / 255).item()
            # ssim_single = 0
           #lpips = ps_loss((sr / 255 * 2 - 1),(hr / 255 * 2 - 1)).item()
           
           total_psnr.append(psnr)
           total_ssim.append(ssim)

           total_ssim_RGB.append(ssim_RGB)
           #total_lpips.append(lpips)
            
          
           #print("PSNR(dB)/SSIM/SSIM_RGB: %.2f/%.4f/%.4f/%.4f." %(psnr, ssim,ssim_RGB,lpips))
           print("PSNR(dB)/SSIM/SSIM_RGB: %.2f/%.4f/%.4f." %(psnr, ssim,ssim_RGB))
      
        print("PSNR: %.2f      SSIM: %.4f     " % (sum(total_psnr)/len(total_psnr),
                                                                  sum(total_ssim)/len(total_ssim),sum(total_ssim_RGB)/len(total_ssim_RGB)))
      


        print("==================================================")
        print("===> Finished !")

if __name__ == '__main__':
    main()

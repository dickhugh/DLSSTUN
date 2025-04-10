'''
test with the trained model
'''
#from dataloader import dataset
from Net import X
from utils import *
#from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import scipy.io as scio
import time
import datetime
import os
import numpy as np
from torch.autograd import Variable

device = torch.device("cpu")
mask_path = "/data/xj_data/RCASSI/real-data/mask/mask_3d_shift_real_256.mat"#mask path#
test_path = "/data/xj_data/RCASSI/test-data/test/" #test data path#
nC = 27
last_train = 500                       
model_save_filename = '2024_11_29_14_49_57'
test_data = LoadTest(test_path)
batch_size = len(test_data)
mask3d_batch = generate_masks(mask_path, batch_size)#mask batch generation#
model = X(nC, 3).to(device)
path_w = r'/data/xuji/xj/X/model/' + model_save_filename + '/log result.txt'

if last_train != 0:
    model = torch.load('./model/' + model_save_filename + '/model_epoch_{}.pth'.format(last_train), map_location=device)
if isinstance(model, nn.DataParallel):
    model=model.module


def test(last_train):
    result=''
    psnr_list, ssim_list = [], []
    test_gt = test_data.to(device).float()
    test_y = gen_meas_torch(test_gt, mask3d_batch)
    begin = time.time()
    with torch.no_grad():
        model_out,_ = model(test_y,mask3d_batch)
    end = time.time()
    for k in range(test_gt.shape[0]):
        psnr_val = torch_psnr(model_out[k,:,:,:], test_gt[k,:,:,:])
        ssim_val = torch_ssim(model_out[k,:,:,:], test_gt[k,:,:,:])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
    pred = torch.squeeze(model_out)
    truth = torch.squeeze(test_gt)
    pred = np.transpose(pred.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(truth.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    print('===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'.format(last_train, psnr_mean, ssim_mean, (end - begin)))
    result=result+'===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'.format(last_train, psnr_mean, ssim_mean, (end - begin))+'\n'
    with open(path_w, "w") as f:
        f.write(result)
    return (pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean)
    
     
def main():
    if model_save_filename == '':
        date_time = str(datetime.datetime.now())
        date_time = time2file_name(date_time)
    else:
        date_time = model_save_filename
    result_path = 'recon' + '/' + date_time
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean) = test(last_train)
    
    name = result_path + '/' + 'test_{}_{:.2f}_{:.3f}'.format(last_train, psnr_mean, ssim_mean) + '.mat'
    scio.savemat(name, {'truth':truth, 'pred': pred, 'psnr_list':psnr_all, 'ssim_list':ssim_all})
        
if __name__ == '__main__':
    main()    
    


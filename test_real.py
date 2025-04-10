'''
Real measurement recovery
---------------------------------------------------------------------------------
Real measurement size in our paper: 768*1024
Use segmentation function to transform the measurement into 12 blocks.(12*256*256)
Use block2img concatenate blocks to a whole HSI
---------------------------------------------------------------------------------
'''
from Net import X
from utils import *
from segmentation import *
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

test_path = "/data/xj_data/RCASSI/real-data/measurement"
mask_path = "/data/xj_data/RCASSI/real-data/mask/mask_3d_shift.mat"
meas_real = Load_real_data(test_path)
mask_3d_shift = Load_mask_data(mask_path)
meas_use = meas_real
mask_use = mask_3d_shift
meas_block = meas_segmentation(meas_use) 
mask_block_batch = mask_segmentation(mask_use)
last_train = 499                     
model_save_filename = '2024_11_29_14_49_57'         
batch_size = 12
nC = 27
model = X(nC, 3)

if last_train != 0:
    model = torch.load('./model/' + model_save_filename + '/model_epoch_{}.pth'.format(last_train))    
if isinstance(model, nn.DataParallel):
    model=model.module
model = model.to(device)


def test(epoch):
    out_list =[]
    model.eval()
    begin = time.time()
    with torch.no_grad():
        for i in range(meas_block.shape[0]):
            model_out,_ = model(meas_block[i,:,:,:], mask_block_batch)
            out_list.append(model_out)
    model_out = torch.stack(out_list, dim=0)
    end = time.time()
    out = block2image(model_out)
    pred = np.transpose(out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    print('===> time: {:.2f}'.format((end - begin)))
    return pred
    
     
def main():
    if model_save_filename == '':
        date_time = str(datetime.datetime.now())
        date_time = time2file_name(date_time)
    else:
        date_time = model_save_filename
    result_path = 'recon' + '/' + date_time
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    epoch = last_train
    pred = test(epoch)
    
    name = result_path + '/' + 'Test_real_{}'.format(last_train) + '.mat'
    scio.savemat(name, {'pred': pred})
        
if __name__ == '__main__':
    main()    
    


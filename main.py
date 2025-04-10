import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#from dataloader import dataset
from Net import X
from utils import *
from tqdm import tqdm
#from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import scipy.io as scio
import time
import datetime
import numpy as np
from torch.autograd import Variable
## activate GPU #

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

data_path = "/data/xj_data/RCASSI/cave-radom-1024/"  #train data path#
real_path = "/data/xj_data/RCASSI/real-data/mask/mask_3d_shift.mat"#real data path#
test_path = "/data/xj_data/RCASSI/test-data/test/" #test data path#

nC, H, W = 27, 256, 256
batch_size = 2
last_train = 0                                           # for finetune
model_save_filename = ''                 # for finetune
max_epoch = 500
learning_rate = 0.0004
epoch_sam_num = 5000
batch_num = int(np.floor(epoch_sam_num/batch_size))
mask_3d_shift = Load_mask_data(real_path)#mask and real measurement#
train_set = LoadTraining(data_path)
test_data = LoadTest(test_path)
batch_size_test = len(test_data)

model = X(nC, 5).cuda()
#model = nn.DataParallel(model)
 
#load model (starts from the last checkpoint)#
if last_train != 0:
    model = torch.load('./model/' + model_save_filename + '/model_epoch_{}.pth'.format(last_train))    
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, betas=(0.9, 0.999))
lf = lossFuc().cuda()

def train(epoch, logger):
    epoch_loss = 0
    begin = time.time()
    for i in tqdm(range(batch_num)):
        gt_batch = shuffle_crop(train_set, batch_size)#train batch preparation#
        gt = Variable(gt_batch).cuda().float()
        multi_Phi = generate_real_multimasks(mask_3d_shift, nC, H, W, batch_size)#mask batch generation#
        y = gen_meas_torch(gt, multi_Phi, is_training = True)#generate measurement#
        optimizer.zero_grad()
        model_out, layers_sym = model(y, multi_Phi)
        Loss = lf(model_out, layers_sym, gt)
        epoch_loss += Loss.data
        Loss.backward()
        optimizer.step()
        # if (i+1)%200 ==0:
        #     print("CheckPoint: {} batchs have been trained".format(i+1))
    end = time.time()
    psnr_list, ssim_list = [], []
    for k in range(gt.shape[0]):#calculate psnr and ssim#
        psnr_val = torch_psnr(model_out[k,:,:,:], gt[k,:,:,:])
        ssim_val = torch_ssim(model_out[k,:,:,:], gt[k,:,:,:])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
    pred = torch.squeeze(model_out)
    truth = torch.squeeze(gt)
    pred = np.transpose(pred.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(truth.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    logger.info("===> Epoch {} Complete: Avg. Loss: {:.6f} time: {:.2f}".format(epoch, epoch_loss/batch_num, (end - begin)))
    logger.info("===> Epoch {}: psnr = {:.2f}, ssim = {:.3f}".format(epoch, psnr_mean, ssim_mean))

#test after every epoch#
def test(epoch, logger):
    psnr_list, ssim_list = [], []
    test_gt = test_data.cuda().float()
    multi_Phi = generate_real_multimasks(mask_3d_shift, nC, H, W, batch_size_test)#mask batch generation
    y = gen_meas_torch(test_gt, multi_Phi, is_training = False)#measurement generation
    model.eval()
    begin = time.time()
    with torch.no_grad():
        model_out, layers_sym = model(y, multi_Phi)
        Loss = lf(model_out, layers_sym, test_gt)
    end = time.time()
    for k in range(test_gt.shape[0]):#calculate psnr and ssim#
        psnr_val = torch_psnr(model_out[k,:,:,:], test_gt[k,:,:,:])
        ssim_val = torch_ssim(model_out[k,:,:,:], test_gt[k,:,:,:])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    logger.info('===> Epoch {}: Avg. Loss: {:.6f}, testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'.format(epoch, Loss, psnr_mean, ssim_mean, (end - begin)))
    model.train()
    return (pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean)

#function for saving model#   
def checkpoint(epoch, model_path, logger):
    model_out_path = './' + model_path + "/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))
     
def main(learning_rate):
    if model_save_filename == '':
        date_time = str(datetime.datetime.now())
        date_time = time2file_name(date_time)
    else:
        date_time = model_save_filename
    result_path = 'recon' + '/' + date_time
    model_path = 'model' + '/' + date_time
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logger = gen_log(model_path)
    logger.info("Learning rate:{}, batch_size:{}.\n".format(learning_rate, batch_size))
    psnr_max = 0
    
    for epoch in range(last_train + 1, last_train + max_epoch + 1):
        train(epoch, logger)
        (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean) = test(epoch, logger)

        if psnr_mean > psnr_max:
            psnr_max = psnr_mean
            if psnr_mean > 27:
                name = result_path + '/' + 'Test_{}_{:.2f}_{:.3f}'.format(epoch, psnr_mean, ssim_mean) + '.mat'
                scio.savemat(name, {'truth':truth, 'pred': pred, 'psnr_list':psnr_all, 'ssim_list':ssim_all})
                checkpoint(epoch, model_path, logger)

        if epoch % 50 == 0:
            name = result_path + '/' + 'Test_{}_{:.2f}_{:.3f}'.format(epoch, psnr_mean, ssim_mean) + '.mat'
            scio.savemat(name, {'truth':truth, 'pred': pred, 'psnr_list':psnr_all, 'ssim_list':ssim_all})
            checkpoint(epoch, model_path, logger)
        #if (epoch % lr_epoch == 0) and (epoch < 200):
            #learning_rate = learning_rate * lr_scale
            #logger.info('Current learning rate: {}\n'.format(learning_rate))

if __name__ == '__main__':
    main(learning_rate)    
    


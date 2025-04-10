import scipy.io as sio
import os 
import numpy as np
# import matplotlib.pyplot as plt
import math
import random
import torch 
import logging
from ssim_torch import ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #when train
#device = torch.device("cpu")                                          #when test

class lossFuc(torch.nn.Module):

    def __init__(self):
        super(lossFuc, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, prediction, prediction_symmetric, gt):

        cost_mean = self.mse(prediction, gt)
        cost_symmetric = 0
        for k in range(len(prediction_symmetric)):
            cost_symmetric += torch.mean(torch.pow(prediction_symmetric[k], 2))

        cost_all = cost_mean + 0.01 * cost_symmetric
        return cost_all

def Load_mask_data(mask_path):
    mask_3d_shift = sio.loadmat(mask_path)['mask_3d_shift']
    mask_3d_shift = np.transpose(mask_3d_shift, [2,0,1])
    mask_3d_shift = torch.from_numpy(mask_3d_shift).to(device).float()
    return mask_3d_shift

def Load_real_data(test_path):
    imgs = []
    scene_list = os.listdir(test_path)
    scene_list.sort()
    for i in range(len(scene_list)): 
        scene_path = test_path + scene_list[i]
        if 'mat' not in scene_path:
            continue
        img_dict = sio.loadmat(scene_path)
        if 'meas_all' in img_dict:
            img = img_dict['meas_all']
        img = img.astype(np.float32)
        imgs.append(img)
    meas_real = np.stack(imgs, axis=0)
    meas_real = torch.from_numpy(meas_real).to(device).float()
    return meas_real

def generate_masks(mask_path, batch_size):
    mask = sio.loadmat(mask_path)
    mask3d_shift = mask['mask_3d_shift_256']
    [H, W, nC] = mask3d_shift.shape
    mask3d_shift = np.transpose(mask3d_shift, [2, 0, 1])
    mask3d_shift = torch.from_numpy(mask3d_shift)
    mask3d_batch = mask3d_shift.expand([batch_size, nC, H, W]).to(device).float()
    return mask3d_batch#bs,27,256,256

def generate_real_multimasks(mask_3d_shift, nC, H, W, batch_size):
    mask_batch = torch.zeros(batch_size, nC, H, W).to(device).float()
    mask_use = mask_3d_shift[:,:,5:-25]
    h, w = mask_use.shape[1], mask_use.shape[2]
    for i in range(batch_size):
        x_index = np.random.randint(0, h - H)
        y_index = np.random.randint(0, w - W)
        mask_small = mask_use[:, x_index:x_index+H, y_index:y_index+W]
        mask_batch[i,:,:,:] = mask_small
    return mask_batch

'''
def generate_masks(mask_path, batch_size):

    mask = sio.loadmat(mask_path)
    mask3d_shift = mask['mask_3d_shift_use']
    [H, W, nC] = mask3d_shift.shape
    mask3d_shift = np.transpose(mask3d_shift, [2, 0, 1])
    mask3d_shift = torch.from_numpy(mask3d_shift)
    mask3d_batch = mask3d_shift.expand([batch_size, 1, nC, H, W]).to(device).float()
    return mask3d_batch#bs,1,27,256,256
'''

def LoadTraining(path):
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    print('training sences:', len(scene_list))
    max_ = 0
    for i in range(len(scene_list)):                          #10
        scene_path = path + scene_list[i]
        if 'mat' not in scene_path:
            continue
        img_dict = sio.loadmat(scene_path)
        if "truth" in img_dict:
            img = img_dict['truth']/65536.
        img = img.astype(np.float32)
        imgs.append(img)
        print('Sence {} is loaded. {}'.format(i, scene_list[i]))

    return imgs

def LoadTest(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), 256, 256, 27))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img = sio.loadmat(scene_path)['matr']
        img[img<0] = 0
        test_data[i,:,:,:] = img
        print(i, img.shape, img.max(), img.min())
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data
    
def psnr(img1, img2):
    psnr_list = []
    for i in range(img1.shape[0]):
        total_psnr = 0
        #PIXEL_MAX = img2.max()
        PIXEL_MAX = img2[i,:,:,:].max()
        for ch in range(27):
            mse = np.mean((img1[i,:,:,ch] - img2[i,:,:,ch])**2)
            total_psnr += 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        psnr_list.append(total_psnr/img1.shape[3])
    return psnr_list

def torch_psnr(img, ref):      
    nC = img.shape[0]
    pixel_max = torch.max(ref)
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i,:,:] - ref[i,:,:]) ** 2)
        psnr += 20 * torch.log10(pixel_max / torch.sqrt(mse))
    return psnr/nC

def torch_ssim(img, ref):   
    return ssim(torch.unsqueeze(img,0), torch.unsqueeze(ref,0))


def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename


def shuffle_crop(train_data, batch_size):
    index = np.random.choice(range(len(train_data)), batch_size)
    processed_data = np.zeros((batch_size, 256, 256, 27), dtype=np.float32)
    for i in range(batch_size):
        h, w, _ = train_data[index[i]].shape
        x_index = np.random.randint(0, h - 256)
        y_index = np.random.randint(0, w - 256)
        gt_img = train_data[index[i]][x_index:x_index + 256, y_index:y_index + 256, :] 
        rot_angle = random.randint(1,4)
        gt_img = np.rot90(gt_img, rot_angle)
        processed_data[i, :, :, :] = gt_img
    gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))
    return gt_batch


def gen_meas_torch(data_batch, mask3d_batch, is_training=True):
    nC = data_batch.shape[1]
    # if is_training is False:
    #     [batch_size, nC, H, W] = data_batch.shape
    #     mask3d_batch = (mask3d_batch[0,:,:,:]).expand([batch_size, nC, H, W]).to(device).float()
    meas = (torch.sum(mask3d_batch*data_batch, 1)/nC*2).to(device).float()          # meas scale
    return meas

#real data process#
def gen_real_phity(meas_batch, mask3d_batch):
    y_temp = (torch.unsqueeze(meas_batch, 1)).repeat(1,27,1,1).to(device)
    PhiTy = torch.mul(y_temp, mask3d_batch).to(device).float()
    return y_temp

def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    
    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO) 
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

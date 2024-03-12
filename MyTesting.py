import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import cv2
from lib.pvt import IdeNet
from utils.dataloader import My_test_dataset 


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size default 352')
parser.add_argument('--pth_path', type=str, default='/dataset2/gjw/IdeNet/checkpoints/IdeNet/Net_epoch_best.pth')
opt = parser.parse_args()
for _data_name in ['CAMO','CHAMELEON','NC4K', 'COD10K']:
    data_path = '/dataset2/gjw/Data/TestDataset/{}/'.format(_data_name)
    save_path = './result/{}/{}/'.format(opt.pth_path.split('/')[-2], _data_name)
    model = IdeNet(train_mode=False)
    model.load_state_dict(torch.load(opt.pth_path, map_location='cuda:0'))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Image/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    print('root',image_root,gt_root)
    test_loader = My_test_dataset(image_root, gt_root, opt.testsize)
    print('****',test_loader.size)
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        print('***name',name)
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        P = model(image)
        P[-1] = (torch.tanh(P[-1]) + 1.0) / 2.0
        
        res = F.upsample(P[-1], size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {}'.format(_data_name, name))
        cv2.imwrite(save_path+name,res*255)

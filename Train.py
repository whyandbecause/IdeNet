import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.pvt import IdeNet
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter, AdaX, AdaXW
import torch.nn.functional as F
import numpy as np
import logging
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import torch.utils.data as data


####
####CUDA_VISIBLE_DEVICES=0 python3 Train.py
####
def load_matched_state_dict(model, state_dict, print_stats=True):
    """
    Only loads weights that matched in key and shape. Ignore other weights.
    """
    num_matched, num_total = 0, 0
    curr_state_dict = model.state_dict()
    for key in curr_state_dict.keys():
        num_total += 1
        if key in state_dict and curr_state_dict[key].shape == state_dict[key].shape:
            curr_state_dict[key] = state_dict[key]
            num_matched += 1
    model.load_state_dict(curr_state_dict)
    if print_stats:
        print(f'Loaded state_dict: {num_matched}/{num_total} matched')


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def val(model, epoch, save_path, writer):
    """
    validation function
    """
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0

        test_loader = test_dataset(image_root=opt.test_path + '/Image/',
                            gt_root=opt.test_path + '/GT/',
                            testsize=opt.trainsize)

        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            P = model(image)
            # res, res1 = model(image)
            # eval Dice
            res = F.upsample(P[-1], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        logging.info(
            '[Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


def train(train_loader, model, optimizer, epoch, test_path):
    total_step = len(train_loader)
    model.train()
    global best
    size_rates = [1]
    loss_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):   
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            # print('this is trainsize',trainsize)
            P = model(images)
            
            # ---- loss function ----
            losses = [structure_loss(out, gts) for out in P]
            loss=0
            gamma=0.2
            for it in range(len(P)):
                loss += (gamma * (it+1)) * losses[it]
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)     
        # ---- train visualization ----
        if i % 60 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}, lr:{:0.7f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record.show(), optimizer.state_dict()['param_groups'][0]['lr']))
            #print(images.shape)
            logging.info('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record.show()))
    # save model
    save_path = opt.save_path
    if epoch % opt.epoch_save == 0:
        torch.save(model.state_dict(), save_path + str(epoch) + 'ESR-PVT.pth')
   
if __name__ == '__main__':

    ##################model_name#############################
    model_name = 'IdeNet'

    ###############################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,default=150, help='epoch number')
    parser.add_argument('--lr', type=float,default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str,default='AdamW', help='choosing optimizer AdamW or SGD')
    parser.add_argument('--augmentation',default=True, help='choose to do random flip rotation')
    parser.add_argument('--batchsize', type=int,default=4, help='training batch size')
    parser.add_argument('--trainsize', type=int,default=384, help='training dataset size,candidate=352,512,704')
    parser.add_argument('--clip', type=float,default=0.5, help='gradient clipping margin')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--decay_rate', type=float,default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,default='/dataset2/gjw/Data/TrainDataset',help='path to train dataset')
    parser.add_argument('--test_path', type=str,default='/dataset2/gjw/Data/TestDataset/COD10K',help='path to testing dataset')
    parser.add_argument('--save_path', type=str,default='./checkpoints/'+model_name+'/')
    parser.add_argument('--epoch_save', type=int,default=1, help='every n epochs to save model')
    opt = parser.parse_args()


    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    logging.basicConfig(filename=opt.save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")


    # ---- build models ----
    torch.cuda.set_device(0)  # set your gpu device
    model = IdeNet().cuda()


    if opt.load is not None:
        pretrained_dict=torch.load(opt.load)
        print('!!!!!!Sucefully load model from!!!!!! ', opt.load)
        load_matched_state_dict(model, pretrained_dict)

    print(model_name)
    print('model paramters',sum(p.numel() for p in model.parameters() if p.requires_grad))

    best = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    elif opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=5e-4, momentum=0.9)
    else:
        optimizer = AdaXW(params, opt.lr)
        #optimizer = torch.optim.Adam(params,lr=0.001,betas=(0.9, 0.999),eps=1e-08,weight_decay=0,amsgrad=False)
        #optimizer = AdaX(params, opt.lr)
    multi_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40, 50], gamma=0.1)
    print(optimizer)
    image_root = '{}/Image/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)
    
    dataset= get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                             augmentation=opt.augmentation)
    
    writer = SummaryWriter(opt.save_path + 'summary')
    size = [384, 512, 704]
    bt = [8, 6, 3]
    print("#" * 20, "Start Training", "#" * 20)
    best_mae = 1
    best_epoch = 0
    index = 0
    for epoch in range(1, opt.epoch):

        opt.trainsize = size[(epoch-1)%3]
        opt.batchsize = bt[(epoch-1)%3]
        
        #adjust_lr(optimizer, opt.lr, epoch, 0.1, 30)
        #opt.trainsize = size[index]
        #opt.batchsize = bt[index]
        print(opt.trainsize,opt.batchsize)
        multi_scheduler.step()
        dataset.init_aug(opt.trainsize, opt.augmentation)
        train_loader = data.DataLoader(dataset=dataset,
                                  batch_size=opt.batchsize,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True)
        train(train_loader, model, optimizer, epoch, opt.save_path)
        if epoch % opt.epoch_save==0:
            val( model, epoch, opt.save_path, writer)

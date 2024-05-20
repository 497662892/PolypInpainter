import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.pvt import PolypPVT
from utils.dataloader import get_updated_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging

import matplotlib.pyplot as plt

def structure_loss(pred, mask, reduction='mean'):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    
    if reduction == 'mean':
        return (wbce + wiou).mean()
    elif reduction == 'none':
        return wbce + wiou
    else:
        raise ValueError("reduction must be mean or none")


def test(model, path):

    data_path = path
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    subfolders = "SUN-SEG" in data_path
    
    model.eval()
    test_loader = test_dataset(image_root, gt_root, subfolders, 352)
    num1 = len(test_loader)
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res, res1  = model(image)
        # eval Dice
        res = F.upsample(res + res1 , size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

    return DSC / num1



def train(train_loader, model, optimizer, epoch, test_path, args):
    model.train()
    global best
    size_rates = [0.75, 1, 1.25] 
    
    loss_P1_record = AvgMeter()
    loss_P2_record = AvgMeter()
    
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
            # print("the shape of images is ", images.shape)
            P1, P2= model(images)
            # ---- loss function ----
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss = loss_P1 + loss_P2 
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_P1_record.update(loss_P1.data, opt.batchsize)
                loss_P2_record.update(loss_P2.data, opt.batchsize)
                
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            logger.info(f'{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], Step [{i:04d}/{total_step:04d}], '
                    f'loss P1: {loss_P1_record.show():.4f}, loss P2: {loss_P2_record.show():.4f}, ')

    # save model 
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # torch.save(model.state_dict(), save_path +str(epoch)+ 'PolypPVT.pth') #no need to save every epoch
    # choose the best model

    global dict_plot
   
    if (epoch + 1) % 1 == 0:

        meandice = test(model, test_path)
        print("the current epoch is {}, the validation dice is {}".format(epoch, meandice))
        if meandice > best:
            best = meandice
            torch.save(model.state_dict(), save_path +str(epoch)+ 'PolypPVT-best.pth')
            logger.info('##############################################################################best:{}'.format(best))


def plot_train(dict_plot=None, name = None):
    color = ['red', 'lawngreen', 'lime', 'gold', 'm', 'plum', 'blue']
    line = ['-', "--"]
    for i in range(len(name)):
        plt.plot(dict_plot[name[i]], label=name[i], color=color[i], linestyle=line[(i + 1) % 2])
        transfuse = {'CVC-300': 0.902, 'CVC-ClinicDB': 0.918, 'Kvasir': 0.918, 'CVC-ColonDB': 0.773,'ETIS-LaribPolypDB': 0.733, 'test':0.83}
        plt.axhline(y=transfuse[name[i]], color=color[i], linestyle='-')
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.title('Train')
    plt.legend()
    plt.savefig('eval.png')
    # plt.show()
    
    
if __name__ == '__main__':
    dict_plot = {'CVC-300':[], 'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[]}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    ##################model_name#############################
    training_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    ###############################################
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--training_type', type=str,
                        default='original', help='the type of model training', choices=['original','basic', 'casual'])
    
    parser.add_argument('--use_background', default=False, 
                        action='store_true', help='whether to use negative_background in training')
    
    parser.add_argument('--align_score_cutoff', type=float,
                        default=0.8, help='the cutoff value for alignment score')
    
    parser.add_argument("--confidence_score_cutoff", type=float, default=1.0, help="the cutoff value for confidence score")
    
    parser.add_argument('--max_aug', type=int,
                        default=None, help='the maximum number of synthetic images')
    
    parser.add_argument('--selection_rule', type=str,
                        default='prediction_score', choices=["prediction_score", "random", "priority"], 
                        help='the selection rule for synthetic images')

    parser.add_argument('--epoch', type=int,
                        default=80, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--switch_ratio', type=float,
                        default=0.0, help='the probability of switching training data color')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='./dataset/TrainDataset/',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='./dataset/TestDataset/',
                        help='path to testing Kvasir dataset')
    
    parser.add_argument('--csv_root', type=str,
                        default='./dataset/TrainDataset/1.csv', help='the path to the csv file of synthetic data')
    
    parser.add_argument("--model_name", type=str, default='PolypPVT')
    
    parser.add_argument('--pretrained_model', type=str, default= None)
    parser.add_argument('--finetune', action='store_true')
    
    parser.add_argument('--train_save', type=str,
                        default='./model_pth/'+  parser.parse_args().model_name +"_"+training_time+'/')
    
    

    opt = parser.parse_args()
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', datefmt='%Y-%m-%d %I:%M:%S %p')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    
    
    if opt.finetune:
        model = PolypPVT()
        model.load_state_dict(torch.load(opt.pretrained_model))
        model.cuda()
    else:
        model = PolypPVT().cuda()

    best = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    if opt.training_type == 'original':
        image_root = '{}/images/'.format(opt.train_path)
        gt_root = '{}/masks/'.format(opt.train_path)
    else:
        image_root = opt.train_path
        gt_root = None
    
    if "SUN-SEG" in opt.train_path:
        subfolders = True
    else:
        subfolders = False

    train_loader = get_updated_loader(image_root, gt_root, csv_root= opt.csv_root, subfolders = subfolders, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation, switch_ratio=opt.switch_ratio, training_type=opt.training_type,
                              align_score_cutoff=opt.align_score_cutoff, max_aug=opt.max_aug, prediction_score_cutoff=opt.confidence_score_cutoff)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, opt.test_path, args = opt)
    
    # plot the eval.png in the training stage
    # plot_train(dict_plot, name)

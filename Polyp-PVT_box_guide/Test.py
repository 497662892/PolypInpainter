import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.pvt import GuidedPolypPVT
from utils.dataloader import test_dataset
import datetime
import cv2
import pandas as pd
import matplotlib.pyplot as plt

def unnorm(img):
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    return img

def generate_grid(image, gt, bbox, res, name, dice, grid_save_path):
    shape = (352, 352)
    
    gt = cv2.resize(gt, (shape[1], shape[0]))
    res = cv2.resize(res, (shape[1], shape[0]))
    
    image = cv2.resize(image, (shape[1], shape[0]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    bbox = cv2.resize(bbox, (shape[1], shape[0]))
    
    gt = np.repeat(gt[:, :, np.newaxis], 3, axis=2)
    bbox = np.repeat(bbox[:, :, np.newaxis], 3, axis=2)
    res = np.repeat(res[:, :, np.newaxis], 3, axis=2)
    
    grid_image = np.hstack((image, gt, bbox, res))
    
    dice = '{:.4f}'.format(dice)
    dice = "".join(dice.split('.'))
    
    cv2.imwrite(grid_save_path + dice+ "_"+ name.split('.')[0] + '_grid.png', grid_image)

    
def get_dice(res, gt):
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
    return dice

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--datasets', type=str, default='ISIC-2017_Test_v2_Data', help='ISIC-2017_Test_v2_Data')
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default='./model_pth/PolypPVT.pth')
    parser.add_argument('--data_path', type=str, default='/data/diffusers/datasets/polyp/')
    parser.add_argument("--train_dataset_name", type=str, default="ISIC")
    parser.add_argument("--save_res", default=False, action="store_true")
    parser.add_argument("--save_grid", default=False, action="store_true")
    
    opt = parser.parse_args()
    model = GuidedPolypPVT()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()
    # for _data_name in ['CVC-ClinicDB/test', 'CVC-EndoSceneStill/test', "PolypGen/test", 'ETIS-LaribPolypDB/test','Kvasir-SEG/test','sunseg_easy_10','sunseg_hard_10']:
    testing_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    dataset_names = opt.datasets.split('+')
    
    for _data_name in dataset_names:

        ##### put data_path here #####
        base_path = opt.data_path
        data_path = os.path.join(base_path, _data_name)
        ##### save_path #####
        save_path = './result_map/{}/PolypPVT_'.format(opt.train_dataset_name)+testing_time+'/{}/'.format(_data_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        res_save_path = './result_map/{}/PolypPVT_'.format(opt.train_dataset_name)+testing_time+'/{}/res/'.format(_data_name)
        if not os.path.exists(res_save_path):
            os.makedirs(res_save_path)
            
        grid_save_path = './result_map/{}/PolypPVT_'.format(opt.train_dataset_name)+testing_time+'/{}/grid/'.format(_data_name)
        if not os.path.exists(grid_save_path):
            os.makedirs(grid_save_path)
        
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        
        subfolders = "SUN-SEG" in _data_name
        
        test_loader = test_dataset(image_root, gt_root, subfolders, 352)
        
        num1 = len(test_loader)
        print('test numbers:', num1)
        
        DSC = 0.0
        dice_list = []
        file_name = []
        
        for i in range(num1):
            image, gt_original, bbox, name = test_loader.load_data()
            gt = np.asarray(gt_original, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            bbox = bbox.cuda()
            P1,P2 = model(image, bbox)
            res = F.upsample(P1+P2, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            
            
            dice = get_dice(res, gt)
            
            file_name.append(name)
            dice_list.append(dice)
            DSC = DSC + dice
            
            # turn image, gt, bbox into original format
            image = (unnorm(image.squeeze().permute(1,2,0).cpu().numpy())*255).astype(np.uint8)
            bbox = (bbox.squeeze().cpu().numpy()*255).astype(np.uint8)
            gt_original = np.asarray(gt_original, np.uint8)
            res = res*255
            
            if opt.save_grid:
                generate_grid(image, gt_original, bbox, res, name, dice, grid_save_path)
            if opt.save_res:
                cv2.imwrite(res_save_path+name, res)
            
            
        data = {'file_name':file_name, 'dice':dice_list}
        data = pd.DataFrame(data)
        data.to_csv(save_path+'dice.csv', index=False)
        print(_data_name, "dice is :", DSC / num1)
        print(_data_name, 'Finish!')

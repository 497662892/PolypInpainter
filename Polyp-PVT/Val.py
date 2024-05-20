import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.pvt import PolypPVT
from utils.dataloader import test_dataset
import cv2
import datetime
import pandas as pd

def unnorm(img):
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    return img

def generate_grid(image, gt, res, name, dice, grid_save_path):
    shape = gt.shape
    image = cv2.resize(image, (shape[1], shape[0]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    
    gt = np.repeat(gt[:, :, np.newaxis], 3, axis=2)
    res = np.repeat(res[:, :, np.newaxis], 3, axis=2)
    
    grid_image = np.hstack((image, gt, res))
    
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
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default='./model_pth/PolypPVT.pth')
    parser.add_argument('--data_path', type=str, default='/data1/mingxiaoli/datasets')
    opt = parser.parse_args()
    model = PolypPVT()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()
    # for _data_name in ['CVC-ClinicDB/test', 'CVC-EndoSceneStill/test', "PolypGen/test", 'ETIS-LaribPolypDB/test','Kvasir-SEG/test','sunseg_easy_10','sunseg_hard_10']:
    testing_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    for _data_name in ['Kvasir-SEG/val']:

        ##### put data_path here #####
        base_path = opt.data_path
        data_path = os.path.join(base_path, _data_name)
        ##### save_path #####
        save_path = './result_map/PolypPVT_'+testing_time+'/{}/'.format(_data_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        res_save_path = './result_map/PolypPVT_'+testing_time+'/{}/res/'.format(_data_name)
        if not os.path.exists(res_save_path):
            os.makedirs(res_save_path)
            
        grid_save_path = './result_map/PolypPVT_'+testing_time+'/{}/grid/'.format(_data_name)
        if not os.path.exists(grid_save_path):
            os.makedirs(grid_save_path)
        
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, 352)
        
        DSC = 0.0
        dice_list = []
        file_name = []
        
        for i in range(num1):
            image, gt_original, name = test_loader.load_data()
            gt = np.asarray(gt_original, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            P1,P2 = model(image)
            res = F.upsample(P1+P2, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            
            dice = get_dice(res, gt)
            
            file_name.append(name)
            dice_list.append(dice)
            DSC = DSC + dice
            
            # turn image, gt, bbox into original format
            image = (unnorm(image.squeeze().permute(1,2,0).cpu().numpy())*255).astype(np.uint8)
            gt_original = np.asarray(gt_original, np.uint8)
            res = res*255
            
            generate_grid(image, gt_original, res, name, dice, grid_save_path)
            cv2.imwrite(res_save_path+name, res)
            
            
        data = {'file_name':file_name, 'dice':dice_list}
        data = pd.DataFrame(data)
        data.to_csv(save_path+'dice.csv', index=False)
        print(_data_name, "dice is :", DSC / num1)
        print(_data_name, 'Finish!')

import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.pvt import PolypPVT
from utils.dataloader import prediction_dataset
import cv2
import datetime
import pandas as pd

def unnorm(img):
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    return img

def generate_grid(image, gt, res, name, dice, grid_save_path):
    shape = (352, 352)
    
    gt = cv2.resize(gt, (shape[1], shape[0]))
    res = cv2.resize(res, (shape[1], shape[0]))
    
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

def get_size(res):
    prob_map = np.where(res>0.5, 1, 0).astype(np.uint8)
    ratio = np.sum(prob_map)/res.shape[0]/res.shape[1]
    
    return prob_map, ratio
    

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
    model = PolypPVT()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()
    
    dataset_names = opt.datasets.split('+')
    # for _data_name in ['CVC-ClinicDB/test', 'CVC-EndoSceneStill/test', "PolypGen/test", 'ETIS-LaribPolypDB/test','Kvasir-SEG/test','sunseg_easy_10','sunseg_hard_10']:
    testing_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
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
        
        num1 = len(os.listdir(image_root))
        test_loader = prediction_dataset(image_root, 352)
        
        DSC = 0.0
        size_list = []
        file_name = []
        
        for i in range(num1):
            image, name, img_size = test_loader.load_data()

            image = image.cuda()
            P1,P2 = model(image)
            res = F.upsample(P1+P2, size=(img_size[1], img_size[0]), mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            prob_map, prob_size = get_size(res)
            # turn image, gt, bbox into original format
            image = (unnorm(image.squeeze().permute(1,2,0).cpu().numpy())*255).astype(np.uint8)

            res = res*255
            prob_map = prob_map*255
            
            file_name.append(name)
            size_list.append(prob_size)
            
            if opt.save_grid:
                generate_grid(image, prob_map, res, name, prob_size, grid_save_path)
                
            if opt.save_res:
                cv2.imwrite(res_save_path+name, res)
            
            
        data = {'file_name':file_name, 'size':size_list}
        data = pd.DataFrame(data)
        data.to_csv(save_path+'size.csv', index=False)
        print(_data_name, 'Finish!')

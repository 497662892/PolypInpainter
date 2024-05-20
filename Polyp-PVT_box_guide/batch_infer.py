import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.pvt import GuidedPolypPVT, PolypPVT
from utils.dataloader import refine_dataset
import datetime
import cv2
import pandas as pd
import matplotlib.pyplot as plt

def unnorm(img):
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    return img

def generate_grid(image, coarse_mask, res, condition_mask, res_2, name, dice_cond, dice_pred, grid_save_path):
    shape = res.shape
    image = cv2.resize(image, (shape[1], shape[0]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    coarse_mask = cv2.resize(coarse_mask, (shape[1], shape[0]))
    
    coarse_mask = np.repeat(coarse_mask[:, :, np.newaxis], 3, axis=2)
    res = np.repeat(res[:, :, np.newaxis], 3, axis=2)
    condition_mask = np.repeat(condition_mask[:, :, np.newaxis], 3, axis=2)
    res_2 = np.repeat(res_2[:, :, np.newaxis], 3, axis=2)
    
    grid_image = np.hstack((image, coarse_mask, res, condition_mask, res_2))
    
    dice_cond = '{:.4f}'.format(dice_cond).split('.')[1]
    dice_pred = '{:.4f}'.format(dice_pred).split('.')[1]

    
    cv2.imwrite(grid_save_path +"/" + "alignment_score/" + dice_cond + "_" + name, grid_image)
    cv2.imwrite(grid_save_path +"/" + "prediction_score/" + dice_pred + "_" + name, grid_image)

    
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
    parser.add_argument('--images_root', type=str, default='/data1/mingxiaoli/datasets/PolypGen/test')
    parser.add_argument('--coarse_mask_root', type=str, default='/data1/mingxiaoli/datasets/PolypGen/test_coarse_mask')
    parser.add_argument('--output_path', type=str, default='./results')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--condition_mask_root', type=str, default='/data1/mingxiaoli/datasets/PolypGen/test_condition_mask')
    parser.add_argument('--pth_path_original', type=str, default='./model_pth/PolypPVT_original.pth')
    
    opt = parser.parse_args()
    model = GuidedPolypPVT()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()
    
    model_2 = PolypPVT()
    model_2.load_state_dict(torch.load(opt.pth_path_original))
    model_2.cuda()
    model_2.eval()
    # for _data_name in ['CVC-ClinicDB/test', 'CVC-EndoSceneStill/test', "PolypGen/test", 'ETIS-LaribPolypDB/test','Kvasir-SEG/test','sunseg_easy_10','sunseg_hard_10']:
    testing_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    for iter in range(1, opt.iters + 1):
        ##### save_path #####
        save_path = opt.output_path 
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        res_save_path = os.path.join(save_path, 'refined_mask_'+testing_time, str(iter))
        if not os.path.exists(res_save_path):
            os.makedirs(res_save_path)
            
        grid_save_path = os.path.join(save_path, 'refined_grid_'+testing_time, str(iter))
        if not os.path.exists(grid_save_path):
            os.makedirs(grid_save_path)
            
        os.makedirs(grid_save_path +"/" + "alignment_score", exist_ok=True)
        os.makedirs(grid_save_path +"/" + "prediction_score", exist_ok=True)
        
        image_root = opt.images_root + "/" + str(iter) + "/"
        coarse_mask_root = opt.coarse_mask_root + "/" + str(iter) + "/"
        condition_mask_root = opt.condition_mask_root + "/" + str(iter) + "/"
        
        num1 = len(os.listdir(coarse_mask_root))
        test_loader = refine_dataset(image_root, coarse_mask_root, condition_mask_root, 352)
        
        file_names = []
        mask_names = []
        
        alignment_score = []
        prediction_score = []
        
        for i in range(num1):
            image, coarse_mask, condition_mask, name = test_loader.load_data()
            condition_mask = np.array(condition_mask, dtype=np.float32)
            condition_mask /= (condition_mask.max() + 1e-8)
            
            # get the prediction of the guided model
            image = image.cuda()
            coarse_mask = coarse_mask.cuda()
            P1,P2 = model(image, coarse_mask)
            res = F.upsample(P1+P2, size=(opt.resolution, opt.resolution), mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            
            dice_cond = get_dice(res, condition_mask)
            
            # geth the prediction of the original model
            P1,P2 = model_2(image)
            res_2 = F.upsample(P1+P2, size=(opt.resolution, opt.resolution), mode='bilinear', align_corners=False)
            res_2 = res_2.sigmoid().data.cpu().numpy().squeeze()
            res_2 = (res_2 - res_2.min()) / (res_2.max() - res_2.min() + 1e-8)
            
            dice_pred = get_dice(res, res_2)
            
            img_name = os.path.join(image_root, name)
            mask_name = os.path.join(coarse_mask_root, name)
            
            file_names.append(img_name)
            mask_names.append(mask_name)
            
            alignment_score.append(dice_cond)
            prediction_score.append(dice_pred)
            # turn image, gt, bbox into original format
            image = (unnorm(image.squeeze().permute(1,2,0).cpu().numpy())*255).astype(np.uint8)
            coarse_mask = (coarse_mask.squeeze().cpu().numpy()*255).astype(np.uint8)
            res = res*255
            res_2 = res_2*255
            condition_mask = condition_mask*255
            
            generate_grid(image, coarse_mask, res, condition_mask, res_2, name, dice_cond, dice_pred, grid_save_path)
            cv2.imwrite(res_save_path+ "/" +name, res)
            
        
        data = {'image': file_names, "mask": mask_names, 'alignment_score': alignment_score, 'prediction_score': prediction_score}
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(save_path, str(iter)+'.csv'))
        print(str(iter) + ', Finish!')

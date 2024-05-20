import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import cv2
import pickle

def switch_color(img1, img2):
    img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2LAB)
    img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2LAB)
    
    mean1, std1 = cv2.meanStdDev(img1)
    mean2, std2 = cv2.meanStdDev(img2)
    
    img1 = img1.astype(np.float32)
    img1 = (img1 - mean1[:,0])/(std1[:,0] + 1e-8) * std2[:,0] + mean2[:,0]
    img1 = np.clip(img1, 0 , 255).astype(np.uint8)
    img1 = cv2.cvtColor(img1, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(img1.astype(np.uint8))
    
def get_bboxs(mask):
    # found contours
    mask = np.array(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 获取外接矩形
    bboxs = []
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        bboxs.append([x, y, x+w, y+h])
    
    return bboxs

def get_newbbox(bbox, img_shape):
    
    left_free = min(bbox[0], 50)
    up_free = min(bbox[1], 50)
    right_free = min(img_shape[1] - bbox[2], 50)
    down_free = min(img_shape[0] - bbox[3], 50)
    
    x1, x2, y1, y2 = bbox[0], bbox[2], bbox[1], bbox[3]
    
    if left_free != 0:
        x1 = bbox[0] - random.randint(0, left_free)
    if right_free != 0:
        x2 = bbox[2] + random.randint(0, right_free)
    if up_free != 0:
        y1 = bbox[1] - random.randint(0, up_free)
    if down_free != 0:
        y2 = bbox[3] + random.randint(0, down_free)
    
    return [x1, y1, x2, y2]
    
def get_newbbox_val(bbox, img_shape):
    
    left_free = min(bbox[0], 50)
    up_free = min(bbox[1], 50)
    right_free = min(img_shape[1] - bbox[2], 50)
    down_free = min(img_shape[0] - bbox[3], 50)
    
    x1, x2, y1, y2 = bbox[0], bbox[2], bbox[1], bbox[3]
    
    x1 = bbox[0] - 0.5*left_free
    x2 = bbox[2] + 0.5*right_free
    y1 = bbox[1] - 0.5*up_free
    y2 = bbox[3] + 0.5*down_free
    
    return [x1, y1, x2, y2]
    
def convert_to_bounding_rect(mask):
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 确保找到至少一个轮廓
    if len(contours) == 0:
        raise ValueError("No contours found in the mask.")
    
    # 获取外接矩形
    bounding_rect_mask = np.zeros_like(mask)
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        
        # 创建一个全零的新mask，与原mask具有相同的维度
        
        
        # 填充外接矩形区域为1（或255，如果你的mask是8位的）
        bounding_rect_mask[y:y+h, x:x+w] = 255  # 或者使用 255 如果你的mask是8位的
    
    bounding_rect_mask = cv2.dilate(bounding_rect_mask, np.ones((30,30), np.uint8), iterations=1)
    bounding_rect_mask = Image.fromarray(bounding_rect_mask)
    
    return bounding_rect_mask

class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, subfolders, trainsize, augmentations, switch_ratio = 0):
        self.trainsize = trainsize
        self.augmentations = augmentations
        print(self.augmentations)
        if subfolders:
            self.images = []
            self.gts = []
            for folder in os.listdir(image_root):
                self.images += [image_root + folder + '/' + f for f in os.listdir(image_root + folder) if f.endswith('.jpg') or f.endswith('.png')]
                self.gts += [gt_root + folder + '/' + f for f in os.listdir(gt_root + folder) if f.endswith('.jpg') or f.endswith('.png')]
        else:
            self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
            self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.filter_files()
        self.size = len(self.images)
        self.switch_ratio = switch_ratio
        if self.augmentations == 'True':
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, interpolation= transforms.InterpolationMode.BILINEAR, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            
        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            

    def __getitem__(self, index):
        
        image = self.rgb_loader(self.images[index])
        if random.random() < self.switch_ratio:
            image2 = self.rgb_loader(random.choice(self.images))
            image = switch_color(image, image2)
            
        gt = self.binary_loader(self.gts[index])
        
        # get a bbox mask
        bboxs = get_bboxs(gt)
        
        if random.random() < 0.8:
            # use a bbox as the guidance
            new_bbox = np.zeros_like(np.array(gt))
            for bbox in bboxs:
                bbox = get_newbbox(bbox, new_bbox.shape)
                new_bbox[bbox[1]:bbox[3]-1, bbox[0]:bbox[2]-1] = 255
            new_bbox = Image.fromarray(new_bbox)
        
        else:
            # use the expanded mask as the guidance
            new_bbox = gt.copy()
            kernel_size = random.randint(20, 80)
            new_bbox = cv2.dilate(np.array(new_bbox), np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
            new_bbox = Image.fromarray(new_bbox)
        
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)
            
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
            
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.gt_transform is not None:
            new_bbox = self.gt_transform(new_bbox)
            
        return image, gt, new_bbox

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, subfolders, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True, augmentation=False,
               switch_ratio=0):

    dataset = PolypDataset(image_root, gt_root, subfolders, trainsize, augmentation, switch_ratio = switch_ratio)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, subfolders, testsize, use_bbox=False):
        self.testsize = testsize
        if subfolders:
            self.images = []
            self.gts = []
            for folder in os.listdir(image_root):
                self.images += [image_root + folder + '/' + f for f in os.listdir(image_root + folder) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp') or f.endswith('.tif')]
                self.gts += [gt_root + folder + '/' + f for f in os.listdir(gt_root + folder) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp') or f.endswith('.tif')]
        
        else:
            self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
            self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
    
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0
        self.use_bbox = use_bbox

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        
        if self.use_bbox:
            bboxs = get_bboxs(gt)
            new_bbox = np.zeros_like(np.array(gt))
            for bbox in bboxs:
                bbox = get_newbbox_val(bbox, new_bbox.shape)
                new_bbox[bbox[1]:bbox[3]-1, bbox[0]:bbox[2]-1] = 255
            new_bbox = Image.fromarray(new_bbox)
            new_bbox = new_bbox.resize((self.testsize, self.testsize), Image.NEAREST)
            new_bbox = self.gt_transform(new_bbox).unsqueeze(0)
        
        else:
            new_bbox = gt.copy()
            kernel_size = 40
            new_bbox = cv2.dilate(np.array(new_bbox), np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
            new_bbox = Image.fromarray(new_bbox)
            new_bbox = new_bbox.resize((self.testsize, self.testsize), Image.NEAREST)
            new_bbox = self.gt_transform(new_bbox).unsqueeze(0)
        
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, new_bbox, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    
    def __len__(self):
        return len(self.images)



class refine_dataset:
    def __init__(self, image_root, gt_root, condition_mask_root, testsize, use_bbox=False, expand = False):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
        self.condition_masks = [condition_mask_root + f for f in os.listdir(condition_mask_root) if f.endswith('.tif') or f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.condition_masks = sorted(self.condition_masks)
        self.use_bbox = use_bbox
        self.expand = expand
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        coarse_mask = self.binary_loader(self.gts[self.index])
        conditional_mask = self.binary_loader(self.condition_masks[self.index])
        
        coarse_mask = coarse_mask.resize((self.testsize, self.testsize), Image.NEAREST)
        # turn the coarse mask into a bbox_mask
        if self.use_bbox:
            coarse_mask = np.array(coarse_mask)
            coarse_mask = convert_to_bounding_rect(coarse_mask)
            
        if self.expand:
            coarse_mask = cv2.dilate(np.array(coarse_mask), np.ones((30,30), np.uint8), iterations=1)
            coarse_mask = Image.fromarray(coarse_mask)
            
        coarse_mask = self.gt_transform(coarse_mask).unsqueeze(0)
        
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, coarse_mask, conditional_mask, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
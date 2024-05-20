import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import cv2
import pickle  
import pandas as pd


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
    

class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, subfolders, trainsize, augmentations, switch_ratio = 0.0):
        self.trainsize = trainsize
        self.augmentations = augmentations
        print(self.augmentations)
        if subfolders:
            self.images = []
            self.gts = []
            for folder in os.listdir(image_root):
                image_subfolder = image_root + folder + '/'
                gt_subfolder = gt_root + folder + '/'
                self.images += [image_subfolder + f for f in os.listdir(image_subfolder) if f.endswith('.jpg') or f.endswith('.png')]
                self.gts += [gt_subfolder + f for f in os.listdir(gt_subfolder) if f.endswith('.jpg') or f.endswith('.png')]
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
                transforms.RandomRotation(90, expand=False, center=None, fill=None),
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
                transforms.Resize((self.trainsize, self.trainsize), interpolation=transforms.InterpolationMode.NEAREST),
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
        
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)
            
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        return image, gt

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
    
class PolypDataser_csv(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, csv_root, trainsize, augmentations, switch_ratio = 0.0, 
                 align_score_cutoff=0.8, prediction_score_cutoff=1.0, max_aug=None):
        
        self.augmentations = augmentations
        print(self.augmentations)
        # load the original images and masks

        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp') or f.endswith('.tif')]
            
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        # read the csv file
        csv_file = pd.read_csv(csv_root)
        # filter out the images with alignment score less than align_score_cutoff
        filter_csv = csv_file[csv_file['alignment_score'] >= align_score_cutoff]
        filter_csv = filter_csv[filter_csv['prediction_score'] <= confidence_threshold]
        
        if not max_aug:
            self.images += filter_csv['image'].tolist()
            self.gts += filter_csv['mask'].tolist()
    
            
        else:
            sample_number = min(max_aug, len(filter_csv))
            sample_csv = filter_csv.sample(sample_number)
            self.images += sample_csv['inpaint_image'].tolist()
            self.gts += sample_csv['inpaint_mask'].tolist()

                
        self.trainsize = trainsize
        
        assert len(self.images) == len(self.gts)
        self.size = len(self.images)
        print("the training size is: ", self.size)
        
        self.switch_ratio = switch_ratio
        if self.augmentations == 'True':
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, expand=False, center=None, fill=None),
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
                transforms.Resize((self.trainsize, self.trainsize), interpolation=transforms.InterpolationMode.NEAREST),
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
            
        gt_path = self.gts[index]
        if gt_path == "negative":
            gt = Image.new('L', (self.trainsize, self.trainsize), (0))
        else:
            gt = self.binary_loader(self.gts[index])
        
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)
            
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        return image, gt

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


class NewPolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, instance_image_root, instance_gt_root, 
                 prior_image_root, prior_gt_root, trainsize, augmentations, switch_ratio = 0.0):
        
        self.trainsize = trainsize
        self.augmentations = augmentations
        self.switch_ratio = switch_ratio
        print(self.augmentations)
        
        self.instance_images = [instance_image_root + f for f in os.listdir(instance_image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.instance_gts = [instance_gt_root + f for f in os.listdir(instance_gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        
        self.prior_images = [prior_image_root + f for f in os.listdir(prior_image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.prior_gts = [prior_gt_root + f for f in os.listdir(prior_gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        
        self.instance_images = sorted(self.instance_images)
        self.instance_gts = sorted(self.instance_gts)
        
        self.prior_images = sorted(self.prior_images)
        self.prior_gts = sorted(self.prior_gts)
        
        self.filter_files()
        
        self.size = len(self.prior_images) # use the same training size
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
        
        instance_image = self.rgb_loader(self.instance_images[index % len(self.instance_images)])
        instance_image_copy = instance_image.copy()
        instance_gt = self.binary_loader(self.instance_gts[index % len(self.instance_gts)])
        
        prior_image = self.rgb_loader(self.prior_images[index % len(self.prior_images)])
        prior_gt = self.binary_loader(self.prior_gts[index % len(self.prior_gts)])
        
        # switch color
        if random.random() < self.switch_ratio:
            instance_image = switch_color(instance_image, prior_image)
            
        if random.random() < self.switch_ratio:
            prior_image = switch_color(prior_image, instance_image_copy)
        
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.img_transform is not None:
            instance_image = self.img_transform(instance_image)
            
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.gt_transform is not None:
            instance_gt = self.gt_transform(instance_gt)
            

        
        seed = np.random.randint(2147483647) # make a seed with numpy generator
        random.seed(seed)
        torch.manual_seed(seed)
        if self.img_transform is not None:
            prior_image = self.img_transform(prior_image)
        
        random.seed(seed)
        torch.manual_seed(seed)
        if self.gt_transform is not None:
            prior_gt = self.gt_transform(prior_gt)
            
        example = {'instance_image': instance_image, 'instance_gt': instance_gt, 'prior_image': prior_image, 'prior_gt': prior_gt}
        
        return example

    def filter_files(self):
        assert len(self.instance_images) == len(self.instance_gts)
        assert len(self.prior_images) == len(self.prior_gts)
        
        instance_images = []
        instance_gts = []
        
        prior_images = []
        prior_gts = []
        
        for instance_img_path, instance_gt_path in zip(self.instance_images, self.instance_gts):
            instance_img = Image.open(instance_img_path)
            instance_gt = Image.open(instance_gt_path)
            
            if instance_img.size == instance_gt.size:
                instance_images.append(instance_img_path)
                instance_gts.append(instance_gt_path)
                
        for prior_img_path, prior_gt_path in zip(self.prior_images, self.prior_gts):
            prior_img = Image.open(prior_img_path)
            prior_gt = Image.open(prior_gt_path)
            
            if prior_img.size == prior_gt.size:
                prior_images.append(prior_img_path)
                prior_gts.append(prior_gt_path)
                
        self.instance_images = instance_images
        self.instance_gts = instance_gts
        
        self.prior_images = prior_images
        self.prior_gts = prior_gts

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


def collate_fn(batch):
    images = []
    gts = []
    
    for sample in batch:
        images.append(sample['instance_image'])
        gts.append(sample['instance_gt'])
        
    for sample in batch:
        images.append(sample['prior_image'])
        gts.append(sample['prior_gt'])
        
    images = torch.stack(images, dim=0)
    gts = torch.stack(gts, dim=0)
    
    return images, gts


def get_loader(image_root, gt_root, subfolders, batchsize, trainsize, shuffle=True, switch_ratio = 0, num_workers=4, pin_memory=True, augmentation=False, sampler = None):
    dataset = PolypDataset(image_root, gt_root, subfolders, trainsize, augmentation, switch_ratio=switch_ratio)
        
    data_loader = data.DataLoader(dataset=dataset,
                                 batch_size=batchsize,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory,
                                 sampler=sampler,
                                 shuffle=shuffle)
    return data_loader

def get_updated_loader(image_root, gt_root, csv_root, subfolders, batchsize, trainsize, shuffle=True, switch_ratio = 0, num_workers=4, pin_memory=True, augmentation=False,
                       sampler = None, training_type = 'original', align_score_cutoff = 0.8, max_aug = None, prediction_score_cutoff=1.0):
    if training_type == 'original':
        dataset = PolypDataset(image_root, gt_root, subfolders, trainsize, augmentation, switch_ratio=switch_ratio)
    else:
        dataset = PolypDataser_csv(image_root, gt_root, csv_root, trainsize, augmentation, switch_ratio=switch_ratio, 
                                     align_score_cutoff=align_score_cutoff,  prediction_score_cutoff=prediction_score_cutoff,max_aug=max_aug)
        
    data_loader = data.DataLoader(dataset=dataset,
                                 batch_size=batchsize,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory,
                                 sampler=sampler,
                                 shuffle=shuffle)
    return data_loader

def get_new_loader(instance_image_root, instance_gt_root, prior_image_root, prior_gt_root, batchsize, 
                   trainsize, shuffle=True, switch_ratio = 0, num_workers=4, pin_memory=True, augmentation=False, sampler = None):

    dataset = NewPolypDataset(instance_image_root, instance_gt_root, prior_image_root, prior_gt_root, trainsize, augmentation, switch_ratio=switch_ratio)
    data_loader = data.DataLoader(dataset=dataset,
                                 batch_size=batchsize,
                                 collate_fn=collate_fn,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory,
                                 sampler=sampler,
                                 shuffle=shuffle
                                 )
    return data_loader

class test_dataset:
    def __init__(self, image_root, gt_root, subfolder, testsize):
        self.testsize = testsize
        
        if subfolder:
            self.images = []
            self.gts = []
            for folder in os.listdir(image_root):
                image_subfolder = image_root + folder + '/'
                gt_subfolder = gt_root + folder + '/'
                self.images += [image_subfolder + f for f in os.listdir(image_subfolder) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
                self.gts += [gt_subfolder + f for f in os.listdir(gt_subfolder)  if f.endswith('.tif') or f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
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

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name
    

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    
    def __len__(self):
        return self.size


class prediction_dataset:
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
        
        self.images = sorted(self.images)
        
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
        img_size = image.size
        image = self.transform(image).unsqueeze(0)
        
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, name, img_size
    

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

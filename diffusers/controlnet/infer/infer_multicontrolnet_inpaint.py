import torch
import torch.utils.data.dataset as dataset
import torch.utils.data.dataloader as dataloader
import torchvision.transforms as transforms
import os
from huggingface_hub import HfApi
from pathlib import Path
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle, random
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline,
    UniPCMultistepScheduler,
)
import argparse


from PIL import Image
import cv2
import argparse
import os
import numpy as np
import pandas as pd
import tqdm
import albumentations as A
import ast
from PIL.Image import Resampling


def random_size(control_image, source_images, mask, ratio = 0.5):
    # reduce the H,W up to 0.5
    ratio = random.uniform(ratio, 1.0)
    print(ratio)
    control_image = control_image.resize((int(512*ratio), int(512*ratio)), Resampling.BILINEAR)
    source_images = source_images.resize((int(512*ratio), int(512*ratio)), Resampling.BILINEAR)
    mask = mask.resize((int(512*ratio), int(512*ratio)), Resampling.BILINEAR)
    
    # padding in right and bottom
    padding = 512 - control_image.size[0]
    control_image = cv2.copyMakeBorder(np.array(control_image), 0, padding, 0, padding, cv2.BORDER_CONSTANT, value = 0)
    source_images = cv2.copyMakeBorder(np.array(source_images), 0, padding, 0, padding, cv2.BORDER_CONSTANT, value = 0)
    mask = cv2.copyMakeBorder(np.array(mask), 0, padding, 0, padding, cv2.BORDER_CONSTANT, value = 0)

    control_image = Image.fromarray(control_image)
    source_images = Image.fromarray(source_images)
    mask = Image.fromarray(mask)
    return control_image, source_images, mask




def switch_color(img1,img2):
    image = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB)
    image2 = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB)

    mean , std  = image.mean(axis=(0,1), keepdims=True), image.std(axis=(0,1), keepdims=True)
    mean2, std2 = image2.mean(axis=(0,1), keepdims=True), image2.std(axis=(0,1), keepdims=True)
    image = np.uint8(np.clip((image-mean)/std*std2+mean2, 0, 255))
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    return image

def update_bbox(bbox, x_ratio, y_ratio):
    bbox = np.array(bbox)
    bbox[0] = int(bbox[0] * x_ratio)
    bbox[1] = int(bbox[1] * y_ratio)
    bbox[2] = int(bbox[2] * x_ratio)
    bbox[3] = int(bbox[3] * y_ratio)
    return bbox

def calculate_masked_rgb_mean(img, mask):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    mean_color = cv2.mean(img, mask=mask)[:3]
    return mean_color

def get_patch_mean(img, i, j, patch_size):
    patch = img[i:i+patch_size, j:j+patch_size]
    return np.mean(patch, axis=(0, 1))

def split_image_and_calculate_means(img, patch_size=32):
    h, w, _ = img.shape
    patch_means = []
    patch_centers = []
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    for i in range(patch_size * 2, h - patch_size * 2, patch_size):
        for j in range(patch_size * 2, w - patch_size * 2, patch_size):
            center_patch_mean = get_patch_mean(img, i, j, patch_size)
            top_patch_mean = get_patch_mean(img, i - patch_size, j, patch_size) if i - patch_size >= 0 else 0
            bottom_patch_mean = get_patch_mean(img, i + patch_size, j, patch_size) if i + patch_size < h else 0
            left_patch_mean = get_patch_mean(img, i, j - patch_size, patch_size) if j - patch_size >= 0 else 0
            right_patch_mean = get_patch_mean(img, i, j + patch_size, patch_size) if j + patch_size < w else 0
            
            # 计算五个区域的平均值
            total_mean = (center_patch_mean + top_patch_mean + bottom_patch_mean + left_patch_mean + right_patch_mean) / 5
            patch_centers.append((i + patch_size // 2, j + patch_size // 2))
            patch_means.append(total_mean)
    
    return patch_means, patch_centers


def find_nearest_patch_center(target_rgb, patch_rgbs, patch_centers):
    target_rgb = np.array(target_rgb)
    patch_rgbs = np.array(patch_rgbs)
    distances = np.linalg.norm(patch_rgbs-target_rgb, axis = 1)
    nearest_idx = np.argmin(distances)
    return patch_centers[nearest_idx]

def recenter_image(img, original_center, new_center):
    # print(original_center, new_center)
    dy, dx = np.subtract(new_center, original_center)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    recentered_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return recentered_img

def get_moments(img):
    img = img.convert('L')
    img = np.array(img)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 使用 max 函数和一个简单的 lambda 函数，找到面积最大的连通区域
    max_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(max_contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cY, cX)

#get the bbox from a mask 
def get_bbox(mask):
    mask = np.array(mask).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    return [x, y, x+w, y+h]

def get_mask(mask, bbox):
    H, W = mask.shape
    extended_bbox=np.array(bbox)
    
    left_freespace=min(bbox[0]-0, 5)
    right_freespace=min(W-bbox[2],5)
    up_freespace=min(bbox[1]-0,5)
    down_freespace=min(H-bbox[3],5)
    
    extended_bbox[0]=bbox[0]-int(left_freespace)
    extended_bbox[1]=bbox[1]-int(up_freespace)
    extended_bbox[2]=bbox[2]+int(right_freespace)
    extended_bbox[3]=bbox[3]+int(down_freespace)
    
    bbox_mask = np.zeros_like(mask)
    bbox_mask[extended_bbox[1]:extended_bbox[3],extended_bbox[0]:extended_bbox[2]] = 1
    mask_img= mask*bbox_mask
    return mask_img

def get_largest_bbox(bboxs):
    bboxs.sort(key = lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
    return bboxs[0]

def copy_paste(original_image, source_images, mask):
    mask = np.where(np.array(mask)> 127,1 ,0).astype(np.uint8)
    mask = mask[:,:,np.newaxis]
    output = original_image.copy()
    output = output*(1 - mask) + source_images*mask
    return Image.fromarray(output)

def get_size(mask):
    mask = cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2GRAY)
    mask = np.where(mask > 0, 1, 0).astype(np.uint8)
    mask_size = np.sum(mask)
    return mask_size/(mask.shape[0]*mask.shape[1])

def get_crop(img, bbox):
    W, H = img.size
    extended_bbox=np.array(bbox)
    
    left_freespace=min(bbox[0]-0, 30)
    right_freespace=min(W-bbox[2],30)
    up_freespace=min(bbox[1]-0,30)
    down_freespace=min(H-bbox[3],30)
    
    extended_bbox[0]=bbox[0]-int(left_freespace)
    extended_bbox[1]=bbox[1]-int(up_freespace)
    extended_bbox[2]=bbox[2]+int(right_freespace)
    extended_bbox[3]=bbox[3]+int(down_freespace)
    
    return img.crop(extended_bbox)

class PolypDataset(dataset.Dataset):
    def __init__(self, original_img_dir, mask_dirs, img_dirs, resolution, iter):
                
        background_path = original_img_dir
        mask_dirs = mask_dirs.split("+")
        img_dirs = img_dirs.split("+")
        
        self.candidate_background = [os.path.join(background_path, background) for background in os.listdir(background_path)]
        
        self.masks = []
        self.source_images = []
        
        for mask_dir, img_dir in zip(mask_dirs, img_dirs):
            masks_path = mask_dir
            source_image_path = img_dir
            
            if "SUN-SEG" in img_dir:
                for folder in os.listdir(source_image_path):
                    masks = os.listdir(os.path.join(masks_path, folder))
                    source_images = os.listdir(os.path.join(source_image_path, folder))
                    self.masks += [os.path.join(masks_path, folder, mask) for mask in masks]
                    self.source_images += [os.path.join(source_image_path, folder, source_image) for source_image in source_images]
            else:
                self.masks += [os.path.join(masks_path, mask) for mask in os.listdir(masks_path)]
                self.source_images += [os.path.join(source_image_path, source_image) for source_image in os.listdir(source_image_path)]
            
        self.masks.sort()
        self.source_images.sort()
        
        assert len(self.masks) == len(self.source_images)
        
        self.iter = iter
        self.resolution = resolution
        self.length = len(self.source_images)
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5], [0.5])
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        
        img_path = self.source_images[index]
        mask_path = self.masks[index]
        name = img_path.split('/')[-1].split('.')[0]
        
        background_path = random.choice(self.candidate_background)

        mask = np.array(load_image(mask_path).convert('L'))
        bbox = get_bbox(mask)
        mask = get_mask(mask, bbox)
        control_image = Image.fromarray(mask.copy()).convert('RGB')
        
        mask = cv2.dilate(mask, np.ones((30,30), np.uint8), iterations=1)
        mask = Image.fromarray(mask)

        
        original_image = load_image(background_path).convert('RGB')
        original_image = original_image.crop((0.05*original_image.size[0], 0.05*original_image.size[1], 0.95*original_image.size[0], 0.95*original_image.size[1]))

        
        source_images = load_image(img_path).convert('RGB')
        bbox_img = get_crop(source_images, bbox)
        bbox_img = bbox_img.resize((512, 512), Image.BILINEAR)
        
        
        mask = mask.resize((512, 512), Image.BILINEAR)
        control_image = control_image.resize((512, 512), Image.BILINEAR)
        original_image = original_image.resize((512, 512), Image.BILINEAR)
        
        # initial_mask to remove the total black area
        initial_mask = (cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2GRAY) > 1).astype(np.uint8)
        
        # # random illumination change
        # random_brightness = A.RandomBrightness(limit=(-0.1, 0.1), p=0.5)
        
        # seed = np.random.randint(0, 2**32 - 1)
        # random.seed(seed)
        # np.random.seed(seed)
        # original_image = random_brightness(image=np.array(original_image))['image']
        # original_image = Image.fromarray(original_image)
        
        # random.seed(seed)
        # np.random.seed(seed)
        # source_images = random_brightness(image=np.array(source_images))['image']
        # source_images = Image.fromarray(source_images)
        
        source_images = source_images.resize((512, 512), Image.BILINEAR)
        source_images_copy = source_images.copy()
        
        temp_size = np.sum(np.array(control_image.convert("L")) > 0)/ 512 / 512
    
        if temp_size > 0.2:
            control_image, source_images, mask = random_size(control_image, source_images, mask, ratio = 0.8)
        elif temp_size > 0.1:
            control_image, source_images, mask = random_size(control_image, source_images, mask, ratio = 0.85)
        elif temp_size > 0.05:
            control_image, source_images, mask = random_size(control_image, source_images, mask, ratio = 0.9)

        old_center = get_moments(control_image)

        mean_of_source_images = calculate_masked_rgb_mean(np.array(source_images), np.where(np.array(mask)> 127,1 ,0).astype(np.uint8))
        original_mean, original_center = split_image_and_calculate_means(np.array(original_image))
        target_center = find_nearest_patch_center(mean_of_source_images, original_mean, original_center)

        source_images = recenter_image(np.array(source_images),old_center, target_center)
        mask = recenter_image(np.array(mask), old_center, target_center)
        control_image = recenter_image(np.array(control_image), old_center, target_center)
        control_image = control_image * initial_mask[:,:,np.newaxis]
        
        if temp_size > 0.025:
            ref_image = original_image
        else:
            ref_image = copy_paste(np.array(original_image), source_images, control_image[:,:,0])
            
        if temp_size > 0.03:
            strength = 0.9
        elif temp_size > 0.02:
            strength = 0.85
        elif temp_size > 0.015:
            strength = 0.7
        elif temp_size > 0.01:
            strength = 0.55
        else:
            strength = 0.5
            
        source_images = Image.fromarray(source_images)
        mask = Image.fromarray(mask)
        control_image = Image.fromarray(control_image)
        
        control_tensor = [control_image, bbox_img]
        
        prompt = "an endoscopic image of a polyp"
        
        example = {
            "normal_image": original_image,
            "background_path": background_path,
            "control_tensor": control_tensor,
            "boundary_control": control_image,
            "surface_control": bbox_img,
            "mask": mask,
            "source_images": source_images_copy,
            "new_center": target_center,
            "prompt": prompt,
            "name": name,
            "ref_image": ref_image,
            "strength": strength,
        }
        return example
               
            
        
def collate_fn(examples):
    normal_img = [example["normal_image"] for example in examples]
    mask = [example["mask"] for example in examples]
    control_tensor = [example["control_tensor"] for example in examples]
    boundary_control = [example["boundary_control"] for example in examples]
    surface_control = [example["surface_control"] for example in examples]
    source_image = [example["source_images"] for example in examples]
    prompt = [example["prompt"] for example in examples]
    name = [example["name"] for example in examples]
    ref_image = [example["ref_image"] for example in examples]
    strength = [example["strength"] for example in examples]
    background_path = [example["background_path"] for example in examples]
    new_center = [example["new_center"] for example in examples]
    
    
    
    return {
        "normal_image": normal_img,
        "mask": mask,
        "control_tensor": control_tensor,
        "boundary_control": boundary_control,
        "surface_control": surface_control,
        "source_image": source_image,
        "prompt": prompt,
        "name": name,
        "ref_image": ref_image,
        "strength": strength,
        "background_path": background_path,
        "new_center": new_center,
    }
    

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--normal_image_dir",
        type=str,
        default="/home/user01/majiajian/data/polyp/negatives",
        help="the path of candidate normal images.",
    )
    parser.add_argument(
        "--polyp_image_dir",
        type=str,
        default="/data1/mingxiaoli/datasets/sun_kvasir_infer/train/images",
        help="The directory where the polyp images are stored.",
    )
    parser.add_argument(
        "--control_image_dir",
        type=str,
        default="/data1/mingxiaoli/datasets/sun_kvasir_infer/train/masks",
        help="The directory where the control images are stored.",
    )
    parser.add_argument(
        "--boundary_controlnet",
        type=str,
        default="/data1/mingxiaoli/datasets/sun_kvasir_infer/train/masks",
        help="The directory where the control images are stored.",
    )
    parser.add_argument(
        "--surface_controlnet",
        type=str,
        default="/data1/mingxiaoli/datasets/sun_kvasir_infer/train/masks",
        help="The directory where the control images are stored.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data1/mingxiaoli/codes/diffusers/examples/controlnet/images",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1, # should be fixed to 1 !!!
        help="The batch size to use for training/validation.", 
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="number of times for data augmentation",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="multicontrolnet_inpaint",
        help="The name of the model to train/evaluate.",
    )
    
    return parser.parse_args(input_args)



def main(args):

    # Set seed for reproducible training
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # create folder for output images
    os.makedirs(os.path.join(args.output_dir, args.model_name), exist_ok=True)
    
    prediction_dir = os.path.join(args.output_dir, args.model_name, "images")
    masks_dir = os.path.join(args.output_dir, args.model_name, "initial_masks")
    conditions_dir = os.path.join(args.output_dir, args.model_name, "conditions")
    grids_dir = os.path.join(args.output_dir, args.model_name, "grids")
    background_path_dir = os.path.join(args.output_dir, args.model_name, "background_path")
    background_dir = os.path.join(args.output_dir, args.model_name, "background")

    
    os.makedirs(prediction_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(grids_dir, exist_ok=True)
    os.makedirs(conditions_dir, exist_ok=True)
    os.makedirs(background_path_dir, exist_ok=True)
    os.makedirs(background_dir, exist_ok=True)

    iterations = 1

    # load the model
    controlnet_1 = args.boundary_controlnet
    controlnet_2 = args.surface_controlnet
    
    stable_diffusion_inpaint = args.pretrained_model_name_or_path
    
    boundary_controlnet = ControlNetModel.from_pretrained(controlnet_1, torch_dtype=torch.float16)
    surface_controlnet = ControlNetModel.from_pretrained(controlnet_2, torch_dtype=torch.float16)
    
    controlnet = [boundary_controlnet, surface_controlnet]
    
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        stable_diffusion_inpaint, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
    )
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    
    generator = torch.manual_seed(args.seed)
    
    for iter in range(iterations):
        iter = args.k # for training using different gpus
        # create dataset
        dataset = PolypDataset(args.normal_image_dir, args.control_image_dir, args.polyp_image_dir, args.resolution, iter)
        
        # create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
        )
        
        num_inference_steps = 50
        names = []
        background_paths = []
        
        for j, batch in enumerate(dataloader):
            ref_image = batch["ref_image"]
            original_background = batch["normal_image"]
            control_tensor = batch["control_tensor"][0]
            boundary_control = batch["boundary_control"]
            surface_control = batch["surface_control"]
            source_images = batch["source_image"]
            mask = batch["mask"]
            name = batch["name"]
            background_path = batch["background_path"]
            prompt = batch["prompt"][0]
            strength = batch["strength"][0]
            # center = batch["new_center"]
            print("prompt: ", prompt)
            print("strength: ", strength)
            
            # sampling from the model
            image = pipe(prompt, num_inference_steps=num_inference_steps, generator=generator, image=ref_image, 
                        mask_image = mask, control_image = control_tensor, strength = strength)[0]

            names += name
            background_paths += background_path
    
            for i in range(len(image)):
                
                mask_temp = mask[i].convert('RGB')
                all_imgs = [original_background[i], surface_control[i], boundary_control[i], mask_temp,  image[i], source_images[i]]
                grid_img = Image.new('RGB', (6 * args.resolution, args.resolution))
                for index, img in enumerate(all_imgs):
                    grid_img.paste(img, (index*args.resolution, 0, index*args.resolution + args.resolution, args.resolution))
                
                os.makedirs(os.path.join(grids_dir, str(iter)), exist_ok=True)
                grid_img.save(os.path.join(grids_dir, str(iter), name[i] +  '.png'))
                
                mask_img =  mask[i]
                os.makedirs(os.path.join(masks_dir, str(iter)), exist_ok=True)
                mask_img.save(os.path.join(masks_dir, str(iter), name[i] + '.png'))
                
                os.makedirs(os.path.join(prediction_dir, str(iter)), exist_ok=True)
                image[i].save(os.path.join(prediction_dir, str(iter), name[i] + '.png'))
                
                boundary_control = boundary_control[i].convert('L')
                os.makedirs(os.path.join(conditions_dir, str(iter)), exist_ok=True)
                boundary_control.save(os.path.join(conditions_dir, str(iter), name[i] + '.png'))
                
                os.makedirs(os.path.join(background_dir, str(iter)), exist_ok=True)
                background_name = background_path[i].split('/')[-1]
                original_background[i].save(os.path.join(background_dir, str(iter), name[i] + "&" + background_name))
        
        pd.DataFrame({"name": names, "background_path": background_paths}).to_csv(os.path.join(background_path_dir, str(iter) + ".csv"), index=False)
                

if __name__ == "__main__":
    args = parse_args()
    main(args)
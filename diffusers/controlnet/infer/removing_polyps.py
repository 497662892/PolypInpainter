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
    StableDiffusionInpaintPipeline,
    UniPCMultistepScheduler,
)
import argparse
from torchvision.utils import make_grid



from PIL import Image
import cv2
import argparse
import os
import numpy as np
import pandas as pd
import tqdm
import albumentations as A





class PolypDataset(dataset.Dataset):
    def __init__(self, data_root, resolution):
        super().__init__()

        mask_root = os.path.join(data_root, "masks")
        images_root = os.path.join(data_root, "images")
        
        if "SUN-SEG" in data_root:
            self.source_images = []
            self.masks = []
            for folder in os.listdir(images_root):
                self.source_images += [os.path.join(images_root, folder, img) for img in os.listdir(os.path.join(images_root, folder))]
                self.masks += [os.path.join(mask_root, folder, img) for img in os.listdir(os.path.join(mask_root, folder))]
        else:
            self.masks = [os.path.join(mask_root, img) for img in os.listdir(mask_root)]
            self.source_images = [os.path.join(images_root, img) for img in os.listdir(images_root)]
        
        self.masks.sort()
        self.source_images.sort()

        self.prompt = "an endoscopic image"
        
        self.resolution = resolution
        self.length = len(self.masks)
        
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        
        img_path = self.source_images[index]
        mask_path = self.masks[index]

        name = img_path.split('/')[-1].split('.')[0]
        
        mask = load_image(mask_path).convert('L')
        control_image = mask.copy().convert('RGB')
        source_images = load_image(img_path).convert('RGB')
        
        mask = mask.resize((512, 512), Image.BILINEAR)
        control_image = control_image.resize((512, 512), Image.BILINEAR)
        source_images = source_images.resize((512, 512), Image.BILINEAR)
        
        mask = cv2.dilate(np.array(mask), np.ones((20,20), np.uint8), iterations=1).astype(np.uint8)
        mask = Image.fromarray(mask)
        
        ref_image = cv2.inpaint(np.array(source_images), np.array(mask).astype(np.uint8), 10, cv2.INPAINT_TELEA)
        ref_image = Image.fromarray(ref_image)
        
        prompt = self.prompt
        
        example = {
            "control_image": control_image,
            "mask": mask,
            "source_images": source_images,
            "prompt": prompt,
            "name": name,
            "ref_image": ref_image,
        }
        return example
               
            
        
def collate_fn(examples):

    mask = [example["mask"] for example in examples]
    control_image = [example["control_image"] for example in examples]
    source_image = [example["source_images"] for example in examples]
    prompt = [example["prompt"] for example in examples]
    name = [example["name"] for example in examples]
    ref_image = [example["ref_image"] for example in examples]
    
    
    return {
        "control_image": control_image,
        "source_image": source_image,
        "mask": mask,
        "prompt": prompt,
        "name": name,
        "ref_image": ref_image,
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
        "--data_dir",
        type=str,
        default="/data1/mingxiaoli/datasets/sun_kvasir_infer/train/images",
        help="The directory where the polyp images are stored.",
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
        default=1,
        help="The batch size to use for training/validation.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="The number of inference steps to use for sampling from the model.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.9,
        help="The strength of the diffusion forward process to use for sampling from the model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="remove_polyp",
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
    masks_dir = os.path.join(args.output_dir, args.model_name, "conditions")
    conditions_dir = os.path.join(args.output_dir, args.model_name, "initial_masks")
    grids_dir = os.path.join(args.output_dir, args.model_name, "grids")

    
    os.makedirs(prediction_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(grids_dir, exist_ok=True)
    os.makedirs(conditions_dir, exist_ok=True)

    # load the model
    stable_diffusion_inpaint = args.pretrained_model_name_or_path
    
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        stable_diffusion_inpaint, torch_dtype=torch.float16, safety_checker=None
    )
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    
    generator = torch.manual_seed(args.seed)
    

    # create dataset
    dataset = PolypDataset(args.data_dir, args.resolution)
    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False
    )
    
    num_inference_steps = args.num_inference_steps
    strength = args.strength
    
    for j, batch in enumerate(dataloader):
        control_image = batch["control_image"]
        source_images = batch["source_image"]
        ref_images = batch["ref_image"]
        name = batch["name"]
        mask = batch["mask"]
        prompt = batch["prompt"]
        print("prompt: ", prompt)
        
        # sampling from the model
        image = pipe(prompt, num_inference_steps=num_inference_steps, generator=generator, image=ref_images, 
                    mask_image = mask, strength = strength)[0]

        for i in range(len(image)):
            
            mask_temp = mask[i].convert('RGB')
            all_imgs = [source_images[i], control_image[i], mask_temp, ref_images[i], image[i]]
            grid_img = Image.new('RGB', (5 * args.resolution, args.resolution))
            for index, img in enumerate(all_imgs):
                grid_img.paste(img, (index*args.resolution, 0, index*args.resolution + args.resolution, args.resolution))
            
            grid_img.save(os.path.join(grids_dir,  name[i] +  '.png'))
            
            mask_img =  mask[i]
            mask_img.save(os.path.join(masks_dir,  name[i] + '.png'))
            
            image[i].save(os.path.join(prediction_dir,  name[i] + '.png'))
            
            control = control_image[i].convert('L')
            control.save(os.path.join(conditions_dir, name[i] + '.png'))
                

if __name__ == "__main__":
    args = parse_args()
    main(args)
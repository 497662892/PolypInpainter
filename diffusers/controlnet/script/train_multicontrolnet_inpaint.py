#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
import pickle, json
import torch.nn as nn

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import cv2
from torch.utils.data import Dataset
import glob

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetInpaintPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
    StableDiffusionPipeline,
)


from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.20.0.dev0")

logger = get_logger(__name__)

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

def switch_color(img1,img2 = None):
    if img2 is None:
        base_paths = [
            "/home/user01/majiajian/data/polyp/negatives",
        ]
        
        base_path = random.choice(base_paths)
        
        img2 = random.choice(os.listdir(base_path))
        img2 = Image.open(os.path.join(base_path, img2)).convert("RGB")
        img2 = np.array(img2)
        
    image = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB)
    image2 = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB)

    mean , std  = image.mean(axis=(0,1), keepdims=True), image.std(axis=(0,1), keepdims=True)
    mean2, std2 = image2.mean(axis=(0,1), keepdims=True), image2.std(axis=(0,1), keepdims=True)
    image = np.uint8(np.clip((image-mean)/std*std2+mean2, 0, 255))
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    return Image.fromarray(image)

def update_bbox(bbox, x_ratio, y_ratio):
    bbox = np.array(bbox)
    bbox[0] = int(bbox[0] * x_ratio)
    bbox[1] = int(bbox[1] * y_ratio)
    bbox[2] = int(bbox[2] * x_ratio)
    bbox[3] = int(bbox[3] * y_ratio)
    return bbox

def calculate_masked_rgb_mean(img, mask):
    mean_color = cv2.mean(img, mask=mask)[:3]
    return mean_color

def split_image_and_calculate_means(img, patch_size = 32):
    h, w, _ = img.shape
    patch_means = []
    patch_centers = []
    
    for i in range(patch_size, h - patch_size, patch_size):
        for j in range(patch_size, w - patch_size, patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]
            patch_mean = np.mean(patch, axis=(0, 1))
            patch_centers.append((i + patch_size // 2, j + patch_size // 2))
            patch_means.append(patch_mean)
            
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

        
# get bbox of a masks
def get_bbox(mask):
    mask = np.array(mask)
    mask = np.where(mask > 127, 1, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 使用 max 函数和一个简单的 lambda 函数，找到面积最大的连通区域
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    return (x, y, x+w, y+h)

# generate  masks
def get_mask(img, bbox, mask):
    W,H = img.size
    
    #update the mask (only in the bbox region)
    mask = np.array(mask)
    new_mask = np.zeros((H,W))
    new_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]]=mask[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    
    kernel_size = random.randint(20, 60)
    mask = cv2.dilate(new_mask, np.ones((kernel_size,kernel_size), np.uint8), iterations=1)

    extended_bbox=np.array(bbox)
    
    left_freespace = bbox[0]
    right_freespace = W - bbox[2]
    up_freespace = bbox[1]
    down_freespace = H - bbox[3]
    
    extended_bbox[0]=bbox[0]-random.randint(0,int(0.1*left_freespace))
    extended_bbox[1]=bbox[1]-random.randint(0,int(0.1*up_freespace))
    extended_bbox[2]=bbox[2]+random.randint(0,int(0.1*right_freespace))
    extended_bbox[3]=bbox[3]+random.randint(0,int(0.1*down_freespace))
    
    mask_img=np.zeros((H,W))
    mask_img[extended_bbox[1]:extended_bbox[3],extended_bbox[0]:extended_bbox[2]]=255
    mask_img=Image.fromarray(mask_img)
    
    crop_img = img.crop(extended_bbox).resize((W,H), Image.Resampling.BILINEAR)
    
    return crop_img, Image.fromarray(mask), Image.fromarray(new_mask), Image.fromarray(new_mask).convert("RGB")

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs

def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds

def log_validation(vae, text_encoder, tokenizer, unet, controlnet, args, accelerator, weight_dtype, step, logging_dir):
    logger.info("Running validation... ")

    controlnet = accelerator.unwrap_model(controlnet)

    pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    validation_images = args.validation_image
    validation_controls = args.validation_controls
    validation_prompts = args.validation_prompts
    validation_backgrounds = args.validation_background_images

    image_logs = []

    iters = 0
    for validation_prompt, validation_image, validation_background, validation_control in zip(validation_prompts, validation_images, validation_backgrounds, validation_controls):
        # load background to inpaint        
        validation_background = Image.open(validation_background).convert("RGB")
        # load the original image of polyp
        validation_image = Image.open(validation_image).convert("RGB")
        # load the mask of polyp
        temp_mask = Image.open(validation_control).convert("L")
        # get the bbox of the polyp
        validation_bbox = get_bbox(temp_mask)
        # get the crop image, inpainting mask, original mask and boundary condition of the polyp
        surface_cond, validation_mask, real_mask , boundary_cond = get_mask(validation_image, validation_bbox, temp_mask)
        
        validation_mask = validation_mask.resize((args.resolution, args.resolution), Image.Resampling.BILINEAR)
        boundary_cond = boundary_cond.resize((args.resolution, args.resolution), Image.Resampling.BILINEAR)
        validation_background = validation_background.resize((args.resolution, args.resolution), Image.Resampling.BILINEAR)
        real_mask = real_mask.resize((args.resolution, args.resolution), Image.Resampling.BILINEAR)
        surface_cond = surface_cond.resize((args.resolution, args.resolution), Image.Resampling.BILINEAR)
        validation_image = validation_image.resize((args.resolution, args.resolution), Image.Resampling.BILINEAR)

        # move the center of the mask to a suitable place of the background
        ref_mean = calculate_masked_rgb_mean(np.array(validation_image), np.where(np.array(validation_mask)>127, 1, 0).astype(np.uint8))
        patch_means, patch_centers = split_image_and_calculate_means(np.array(validation_background))
        new_center = find_nearest_patch_center(ref_mean, patch_means, patch_centers)
        old_center = get_moments(real_mask)

        validation_image = Image.fromarray(recenter_image(np.array(validation_image), old_center, new_center))
        boundary_cond = Image.fromarray(recenter_image(np.array(boundary_cond), old_center, new_center))
        validation_mask = Image.fromarray(recenter_image(np.array(validation_mask), old_center, new_center))
        real_mask = Image.fromarray(recenter_image(np.array(real_mask), old_center, new_center))

        np_validation_mask = np.array(validation_mask).astype(np.uint8)
        np_validation_mask = np.where(np_validation_mask > 127, 1, 0).astype(np.uint8)
        np_validation_mask = np_validation_mask[:, :, np.newaxis]
        validation_background_np = np.array(validation_background).astype(np.uint8)
        validation_background_np = validation_background_np * (1 - np_validation_mask)
        validation_masked_background = Image.fromarray(validation_background_np)

        new_validation_control = [boundary_cond, surface_cond]
        
        with torch.autocast("cuda"):
            images = pipeline(
                prompt = validation_prompt, image = validation_background, mask_image = validation_mask, control_image = new_validation_control, num_inference_steps=50, 
                generator=generator, num_images_per_prompt = args.num_validation_images, strength = 0.85
            ).images

        # images += image
        validation_prompt += "_{}".format(iters)
        image_logs.append(
            {"background_image": validation_background, "validation_image": validation_image, "validation_mask": validation_mask, "boundary_cond": boundary_cond, 
             "surface_cond": surface_cond,"validation_masked_background": validation_masked_background, "images": images, "validation_prompt": validation_prompt}
        )
        iters += 1

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_background = log["background_image"]
                validation_image = log["validation_image"]
                validation_boundary_cond = log["boundary_cond"]
                validation_surface_cond = log["surface_cond"]
                validation_masked_background = log["validation_masked_background"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_background))
                formatted_images.append(np.asarray(validation_image))
                formatted_images.append(np.asarray(validation_boundary_cond))
                formatted_images.append(np.asarray(validation_surface_cond))
                formatted_images.append(np.asarray(validation_masked_background))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
                
                image_grids = image_grid([Image.fromarray(img) for img in formatted_images], 1, len(formatted_images))
                os.makedirs(os.path.join(logging_dir, "validation"), exist_ok=True)
                image_grids.save(os.path.join(logging_dir, "validation", f"{str(step)}-{validation_prompt}.png"))
                
        elif tracker.name == "wandb":
            raise NotImplementedError("Wandb logging is not implemented for validation.")

        return image_logs


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- controlnet
inference: true
---
    """
    model_card = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


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
        "--boundary_controlnet",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights will raise error.",
    )
    parser.add_argument(
        "--surface_controlnet",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights will raise error.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
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
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_root",
        type=str,
        default=None,
        help=(
            "the path of training images. "
        ),
    )
    parser.add_argument(
        "--instance_mask_root",
        type=str,
        default=None,
        help=(
            "the path of training masks. "
        ),
    )
    parser.add_argument(
        "--instance_prompts",
        type=str,
        default=None,
        help=(
            "the prompt of training images. "
        ),
    )
    parser.add_argument(
        "--subfolders",
        action="store_true",
        help=(
            "Whether the images are stored in subfolders. "
        ),
        default=False
    )
    parser.add_argument(
        "--tokenizer_max_length",
        type=int,
        default=None,
        required=False,
        help="The maximum length of the tokenizer. If not set, will default to the tokenizer's max length.",
    )
    parser.add_argument(
        "--text_encoder_use_attention_mask",
        action="store_true",
        required=False,
        help="Whether to use attention mask for the text encoder",
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_list",
        type=str,
        default=None,
        help=(
            "the path of a json file containing the list of validation settings."
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_control",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


class ControlNetDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_mask_root,
        instance_prompts,
        subfolders,
        tokenizer,
        size=512,
        center_crop=False,
        encoder_hidden_states=None,
        tokenizer_max_length=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length

        if subfolders:
            self.images = []
            self.masks = []
            for folder in os.listdir(instance_data_root):
                self.images.extend(glob.glob(os.path.join(instance_data_root, folder, "*.jpg")))
                self.masks.extend(glob.glob(os.path.join(instance_mask_root, folder, "*.png")))
        else:
            self.images = glob.glob(os.path.join(instance_data_root, "*.jpg"))
            self.masks = glob.glob(os.path.join(instance_mask_root, "*.png"))
        
        self.images.sort()
        self.masks.sort()
        assert len(self.images) == len(self.masks)
        
        self.instance_prompts = instance_prompts # a string
        
        self._length = len(self.images)

        self.transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomRotation(degrees=90, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
            ]
        )
        
        self.normalize = transforms.Normalize([0.5], [0.5])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.images[index])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        
        instance_mask = Image.open(self.masks[index]).convert("L")
        
        # logger.info("\n")
        # logger.info(f"Processing image {self.images[index]}")
        # logger.info(f"Processing mask {self.masks[index]}")
        
        instance_bbox = get_bbox(instance_mask)
        bbox_image, instance_mask, real_mask , instance_control = get_mask(instance_image, instance_bbox, instance_mask)
            
        if random.random() < 0.3:
            instance_image = switch_color(np.array(instance_image))
            
        if random.random() < 0.3:
            bbox_image = switch_color(np.array(bbox_image))
            
        temp_seed = np.random.randint(0, 10000000)
        np.random.seed(temp_seed)
        torch.manual_seed(temp_seed)
        image = self.normalize(self.transforms(instance_image))
        
        np.random.seed(temp_seed)
        torch.manual_seed(temp_seed)
        mask = self.transforms(instance_mask)

        np.random.seed(temp_seed)
        torch.manual_seed(temp_seed)
        real_mask = self.transforms(real_mask)
        
        np.random.seed(temp_seed)
        torch.manual_seed(temp_seed)
        conditioning_image = self.transforms(instance_control)
        
        bbox_image = self.transforms(bbox_image)
        
        
        real_mask = torch.where(real_mask>0.5,1,0)
        
        # the outside of the image
        mask = torch.where(mask>0.5,1,0)
        masked_image =  (1 - mask) * image
        
        example["instance_pixel_values"] = image
        example["instance_real_masks"] = real_mask
        example["instance_masked_pixel_values"] = masked_image
        example["instance_mask"] = mask
        example["instance_boundary"] = conditioning_image
        example["instance_surface"] = bbox_image
        
        instance_prompt = self.instance_prompts
        
        text_inputs = tokenize_prompt(
                self.tokenizer, instance_prompt, tokenizer_max_length=self.tokenizer_max_length
            )
        example["instance_prompt_ids"] = text_inputs.input_ids
        example["instance_attention_mask"] = text_inputs.attention_mask
            
        return example


def collate_fn(examples):

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_pixel_values"] for example in examples]
    masked_pixel_values = [example["instance_masked_pixel_values"] for example in examples]
    condition_boundary = [example["instance_boundary"] for example in examples]
    mask = [example["instance_mask"] for example in examples]
    attention_mask = [example["instance_attention_mask"] for example in examples]
    condition_surface = [example["instance_surface"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    masked_pixel_values = torch.stack(masked_pixel_values)
    masked_pixel_values = masked_pixel_values.to(memory_format=torch.contiguous_format).float()
    
    mask = torch.stack(mask)
    mask = mask.to(memory_format=torch.contiguous_format).float()
    
    condition_boundary = torch.stack(condition_boundary)
    condition_boundary = condition_boundary.to(memory_format=torch.contiguous_format).float()
    
    condition_surface = torch.stack(condition_surface)
    condition_surface = condition_surface.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)

    batch = {
        "pixel_values": pixel_values,
        "masked_pixel_values": masked_pixel_values,
        "conditioning_boundary": condition_boundary,
        "conditioning_surface": condition_surface ,
        "input_ids": input_ids,
        "masks": mask,
        "attention_mask": attention_mask,
    }

    return batch



def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
    # prepare the validation settings
        
    
    if args.validation_list is not None:
        validation_list = json.load(open(args.validation_list, 'r'))
        args.validation_background_images = validation_list["validation_background"]
        args.validation_image = validation_list["validation_image"]
        args.validation_controls = validation_list["validation_mask"]
        args.validation_prompts = validation_list["validation_prompt"]
        args.num_validation_images = validation_list["num_validation_images"]


    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    if args.boundary_controlnet:
        logger.info("Loading existing controlnet weights")
        controlnet_1 = ControlNetModel.from_pretrained(args.boundary_controlnet)
    else:
        raise ValueError("Please provide the boundary controlnet model")
    if args.surface_controlnet:
        logger.info("Loading existing controlnet weights")
        controlnet_2 = ControlNetModel.from_pretrained(args.surface_controlnet)
    else:
        raise ValueError("Please provide the surface controlnet model")

    # the first should be the boundary and the second should be the surface condition
    controlnet = MultiControlNetModel([controlnet_1, controlnet_2])
    
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, _, output_dir):
            model = models[0]  # Assuming MultiControlNetModel is the first model
            # logger.info(model)
            boundary_controlnet = model.nets[0]
            surface_controlnet = model.nets[1]
            boundary_controlnet.save_pretrained(os.path.join(output_dir, "boundary_controlnet"))
            surface_controlnet.save_pretrained(os.path.join(output_dir, "surface_controlnet"))

        def load_model_hook(models, input_dir):
            model = models[0]  # Assuming MultiControlNetModel is the first model
            # Assuming that MultiControlNetModel.from_pretrained() is similarly adapted
            # to load each controlnet from its respective directory
            boundary_controlnet = ControlNetModel.from_pretrained(input_dir, subfolder="boundary_controlnet")
            boundary_controlnet.register_to_config(**boundary_controlnet.config)
            
            surface_controlnet = ControlNetModel.from_pretrained(input_dir, subfolder="surface_controlnet")
            surface_controlnet.register_to_config(**surface_controlnet.config)
            
            load_model = MultiControlNetModel([boundary_controlnet, surface_controlnet])
            model.load_state_dict(load_model.state_dict())
            del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()

        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()


    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = controlnet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    

    train_dataset = ControlNetDataset(args.instance_data_root, 
                                      args.instance_mask_root, 
                                      args.instance_prompts,
                                      args.subfolders,
                                      tokenizer,
                                      size = args.resolution,
                                      center_crop=args.center_crop,
                                      )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompts")
        tracker_config.pop("validation_image")
        tracker_config.pop("validation_controls")
        tracker_config.pop("validation_background_images")
        tracker_config.pop("num_validation_images")
        
        tracker_config.pop("instance_prompts")
        tracker_config.pop("instance_data_root")
        tracker_config.pop("instance_mask_root")
        
        

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                masked_latents = vae.encode(batch["masked_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                masked_latents = masked_latents * vae.config.scaling_factor
                
                masks = batch["masks"].to(dtype=weight_dtype)
                masks = torch.nn.functional.interpolate(masks, size=(args.resolution // 8, args.resolution // 8)) 
                
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                latent_model_input = torch.cat([noisy_latents, masks, masked_latents], dim=1)

                # Get the text embedding for conditioning
                encoder_hidden_states = encode_prompt(
                        text_encoder,
                        batch["input_ids"],
                        batch["attention_mask"],
                        text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
                    )
                
                controlnet_image = [batch["conditioning_boundary"].to(dtype=weight_dtype), batch["conditioning_surface"].to(dtype=weight_dtype)]

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                    conditioning_scale=[1.0,1.0]
                )

                # Predict the noise residual
                with torch.autocast("cuda"):
                    model_pred = unet(
                        latent_model_input,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=[
                            sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                        ],
                        mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            controlnet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            logging_dir
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = accelerator.unwrap_model(controlnet)
        boundary_control = controlnet.nets[0]
        surface_control = controlnet.nets[1]
        boundary_control.save_pretrained(os.path.join(args.output_dir, "boundary_controlnet"))
        surface_control.save_pretrained(os.path.join(args.output_dir, "surface_controlnet"))

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)

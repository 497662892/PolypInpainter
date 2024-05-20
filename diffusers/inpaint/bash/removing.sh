#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export MODEL_NAME="/home/user01/majiajian/pretrain_model/stable-diffusion-inpainting"
export OUTPUT_DIR="/home/user01/majiajian/code/diffusion/diffusers/output_model/sd_inpaint/remove_1e5"
export HF_HUB_OFFLINE=1
export validation_list="/home/user01/majiajian/code/diffusion/diffusers/examples/inpaint/concept_list/0430/removing_validation_list_0430.json"

accelerate launch scripts/polyp/train_inpaint.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --resume_from_checkpoint="latest" \
  --instance_data_dir="/home/user01/majiajian/data/polyp/negatives" \
  --instance_mask_dir="/home/user01/majiajian/data/polyp/SUN-SEG_10/TrainDataset_10/masks" \
  --instance_prompt="an endoscopic image" \
  --subfolders \
  --remove=True \
  --validation_list=$validation_list \
  --output_dir=$OUTPUT_DIR \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=5000 \
  --checkpointing_steps=1000 \
  --mixed_precision="fp16" \
  --validation_steps=200 \
  --enable_xformers_memory_efficient_attention \
  --seed=42 

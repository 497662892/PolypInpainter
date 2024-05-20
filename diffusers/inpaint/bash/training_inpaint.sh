#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
# the path of the pretrain model
export MODEL_NAME="/home/user01/majiajian/pretrain_model/stable-diffusion-inpainting"
# the path of the output model
export OUTPUT_DIR="/home/user01/majiajian/code/diffusion/diffusers/output_model/sd_inpaint/inpaint_1e5"
export HF_HUB_OFFLINE=1
# the path of the validation list
export validation_list="/home/user01/majiajian/code/diffusion/diffusers/examples/inpaint/concept_list/validation_list.json"

accelerate launch scripts/train_inpaint.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --resume_from_checkpoint="latest" \
  --instance_data_dir="/home/user01/majiajian/data/polyp/Kvasir-SEG/trainval/images" \
  --instance_mask_dir="/home/user01/majiajian/data/polyp/Kvasir-SEG/trainval/masks" \
  --instance_prompt="an endoscopic image of polyp" \
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

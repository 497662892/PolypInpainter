export CUDA_VISIBLE_DEVICES=5
export MODEL_DIR="/home/user01/majiajian/code/diffusion/diffusers/output_model/sd_inpaint/inpaint_1e5"
export BOUNDARY_CONTROLNET="/home/user01/majiajian/code/diffusion/diffusers/output_model/multicontrolnet/checkpoint-16000/boundary_controlnet"
export SURFACE_CONTROLNET="/home/user01/majiajian/code/diffusion/diffusers/output_model/multicontrolnet/checkpoint-16000/surface_controlnet"
export OUTPUT_DIR="/home/user01/majiajian/data/polyp/Kvasir-SEG"

# model dir is the path of the finetuned stable diffusion backbone model
# boundary_controlnet is the path of the boundary controlnet model (seg)
# surface_controlnet is the path of the surface controlnet model (shuffle)
# output_dir is the path of the output directory

# normal_image_dir is the path of the negative background
# polyp_image_dir is the path of the polyp images
# control_image_dir is the path of the mask of the polyp
# model_name is the name of the model
# resolution is the resolution of the image
# batch_size is the batch size, should be fixed to 1
# k is the index of current augmentation times


accelerate launch infer/infer_multicontrolnet_inpaint.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --boundary_controlnet=$BOUNDARY_CONTROLNET \
 --surface_controlnet=$SURFACE_CONTROLNET \
 --normal_image_dir="/home/user01/majiajian/data/polyp/negatives" \
 --polyp_image_dir="/home/user01/majiajian/data/polyp/Kvasir-SEG/trainval/images" \
 --control_image_dir="/home/user01/majiajian/data/polyp/Kvasir-SEG/trainval/masks" \
 --output_dir=$OUTPUT_DIR \
 --model_name="multiple_controlnet_inpaint" \
 --resolution=512 \
 --batch_size=1 \
 --k=5 \
 --seed=45

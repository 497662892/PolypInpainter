export CUDA_VISIBLE_DEVICES=5
export MODEL_DIR="/home/user01/majiajian/code/diffusion/diffusers/output_model/sd_inpaint/remove_1e5"
export OUTPUT_DIR="/home/user01/majiajian/data/polyp/Kvasir-SEG/remove"

accelerate launch infer/removing_polyps.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --data_dir="/home/user01/majiajian/data/polyp/Kvasir-SEG/trainval" \
 --output_dir=$OUTPUT_DIR \
 --model_name="remove" \
 --resolution=512 \
 --batch_size=10 \
 --strength=0.9 \
 --seed=42 

export CUDA_VISIBLE_DEVICES=1

python -W ignore Train.py \
--switch_ratio 0.0 \
--model_name Kvasir-SEG_guided \
--train_path /home/user01/majiajian/data/polyp/Kvasir-SEG/train \
--test_path /home/user01/majiajian/data/polyp/Kvasir-SEG/val \
--augmentation True 
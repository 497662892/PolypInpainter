export CUDA_VISIBLE_DEVICES=2

# train_path: the path of the training data
# test_path: the path of the validation data
# model_name: the name of the model
# augmentation: whether to use data augmentation

python -W ignore Train.py \
--train_path /home/user01/majiajian/data/polyp/Kvasir-SEG/train \
--test_path /home/user01/majiajian/data/polyp/Kvasir-SEG/val \
--model_name SUN-SEG_baseline \
--augmentation True \
--lr=1e-4
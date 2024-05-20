export CUDA_VISIBLE_DEVICES=4

# train_path: the path of the training data
# test_path: the path of the validation data
# training_type: the type of basic
# align_score_cutoff: the threshold of the alignment score >= is included
# prediction_score_cutoff: the threshold of the prediction score <= is included
# max_aug: the maximum number of augmentations
# model_name: the name of the model
# --fine_tune: whether to fine tune the model
# --pretrained_model: the path of the pretrained model (needed if fine_tune is True)

python -W ignore Train_new.py \
--train_path /home/user01/majiajian/data/polyp/Kvasir-SEG/train \
--test_path /home/user01/majiajian/data/polyp/Kvasir-SEG/val \
--csv_root /home/user01/majiajian/data/polyp/Kvasir-SEG/multiple_controlnet_inpaint/1.csv \
--training_type basic \
--align_score_cutoff 0.8 \
--prediction_score_cutoff 1.0 \
--max_aug None \
--model_name Kvasir-SEG_Aug \
--augmentation True \
--lr=1e-4
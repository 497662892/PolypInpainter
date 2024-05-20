export CUDA_VISIBLE_DEVICES=2

# --datasets: the name of the dataset
# --train_dataset_name: the name of the training dataset
# --save_grid: save the grid for visualization
# --save_res: save the result in csv
# --data_path: the base root of the data
# --pth_path: the path of the model is going to be tested

python -W ignore Test.py \
--datasets 'SUN-SEG_10_test+Kvasir-SEG+CVC-ClinicDB+CVC-300+ETIS-LaribPolypDB' \
--train_dataset_name polyp \
--save_grid \
--save_res \
--data_path /home/user01/majiajian/data/polyp/ \
--pth_path /home/user01/majiajian/code/segmentation/Polyp-PVT/model_pth/Kvasir-SEG/14PolypPVT-best.pth
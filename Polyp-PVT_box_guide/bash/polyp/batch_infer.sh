export CUDA_VISIBLE_DEVICES=7


# --images_root: the path of the images
# --coarse_mask_root: the path of the coarse masks
# --output_path: the path of the output (synthesized images)
# --condition_mask_root: the path of the condition masks
# --resolution: the resolution of the image
# --iters: the number of times for generation
# --testsize: the size of the test
# --pth_path: the path of the psudomask-refinement model
# --pth_path_original: the path of the original model, which is only trained by Kvasir-SEG dataset

python -W ignore batch_infer.py \
--images_root "/home/user01/majiajian/data/polyp/Kvasir-SEG/multiple_controlnet_inpaint/images" \
--coarse_mask_root "/home/user01/majiajian/data/polyp/Kvasir-SEG/multiple_controlnet_inpaint/initial_masks" \
--output_path "/home/user01/majiajian/data/polyp/Kvasir-SEG/multiple_controlnet_inpaint" \
--condition_mask_root "/home/user01/majiajian/data/polyp/Kvasir-SEG/multiple_controlnet_inpaint/conditions" \
--resolution 512 \
--iters 5 \
--testsize 352 \
--pth_path "/home/user01/majiajian/code/segmentation/Polyp-PVT_box_guide/model_pth/Kvasir-SEG_guided/6PolypPVT-best.pth" \
--pth_path_original "/home/user01/majiajian/code/segmentation/Polyp-PVT/model_pth/Kvasir-SEG_baseline/26PolypPVT-best.pth" 
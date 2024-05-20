## PolypInpainter



## 🔆 Introduction
This is a PyTorch implementation of the paper **"Generalize Polyp Segmentation via Inpainting across Diverse Backgrounds and Pseudo-Mask Refinement"** at IEEE ISBI-2024. In this paper, we proposed an inpainting based data augmentation method that can significantly enchance the generalization of polyp segmentation models.

The overview of our data augmentation pipeline is shown below:
![image](https://github.com/497662892/PolypInpainter/blob/main/imgs/pipeline.png)
which includes 5 different components: **Training Inpainting Model**, **Generating Inpainting Images**, **Pseudo-mask Refinement**, **Suitible cases selection**, and **Training Segmentation Model**.

The framework of our proposed data augmentation method is shown below:
![image](https://github.com/497662892/PolypInpainter/blob/main/imgs/inpaint_model.PNG)
![image](https://github.com/497662892/PolypInpainter/blob/main/imgs/refinement.png)

## Checkpoints
The checkpoints we use in this project are available at baidupan:

Link：https://pan.baidu.com/s/1Ds-nRmxXG-45C228rJc56g?pwd=tnoc 
Password：tnoc 
- "checkpoints/inpaint" contains the finettuned model for polyp inpainting
- "checkpoints/remove" contains the finettuned model for polyp removal
- "checkpoints/controlmodules" contains the finettuned model for the multi-controlnet modules


## Environment
please run the following code for environment setup:
```bash
git clone https://github.com/497662892/PolypInpainter.git
cd PolypInpainter
pip install -r requirements.txt
```


## Training Polyp Inpainting Model

### Preparations

Before running the training code, please make sure you have downloaded the pretrained [stable diffusion inpaint 1.5](https://huggingface.co/runwayml/stable-diffusion-inpainting).

You also need to update the concept list for validation via "diffuser/inpaint/concept_list/make_concept_list.ipynb".

Please also update the path in the **"diffuser/inpaint/bash/training_inpaint.sh"** file.

### Training

To train the inpainting model, you can run the following command:
```bash
cd diffuser/inpaint
nohup bash bash/training_inpaint.sh  > "your training log path" &
```

## Training Polyp Remove Model

### Preparations

Please update the concept list for validation via "diffuser/inpaint/concept_list/make_concept_list_negative_only.ipynb".

Please also update the path in the **"diffuser/inpaint/bash/training_remove.sh"** file.

### Training

To train the inpainting model, you can run the following command:
```bash
cd diffuser/inpaint
nohup bash bash/training_remove.sh  > "your training log path" &
```

## Training Controlnet Model

### Preparations

Before running the training code, please make sure you have downloaded the pretrained model [control_v11p_sd15_seg](https://huggingface.co/lllyasviel/control_v11p_sd15_seg) for boundary control and [control_v11e_sd15_shuffle](https://huggingface.co/lllyasviel/control_v11e_sd15_shuffle) for surface control.

You also need to update the concept list for validation via "diffuser/controlnet/concept_list/make_concept_list.ipynb".

Please also update the path in the **"diffuser/controlnet/bash/train/train_multicontrolnet.sh"** file.


### Training
To train the controlnet model, you can run the following command:
```bash
cd diffuser/controlnet
nohup bash bash/train/train_multicontrolnet.sh  > "your training log path" &
```

## Generating Removed Poylp Images

To generate removed polyp images, you need to modified the path in the file "diffuser/controlnet/bash/infer/infer_remove.sh".

Then you can run the command below to generate removed polyp images:
```bash
cd diffuser/controlnet
nohup bash bash/infer/infer_remove.sh  > "your generating log path" &
```

## Generating Inpainting Images

To generate inpainting images, you need to modified the path in the file "diffuser/controlnet/bash/infer/infer_multiplecontrolnet.sh".

Then you can run the command below to generate inpainting images:
```bash
cd diffuser/controlnet
nohup bash bash/infer/infer_multiplecontrolnet.sh  > "your generating log path" &
```


## Training Pseudo-mask Refinement network

### Preparations

Before running the training code, you need to download the pretrained model from [google drive](https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV), to the path of "Polyp-PVT_box_guide/pretrained_pth".

### Training

To train the pseudo-mask refinement network, you can run the following command, after changing the log path in the **"train.sh"** file:
```bash
cd Polyp-PVT_box_guide
nohup bash bash/polyp/train.sh  > "your training log path" &
```

## Pseudo-mask Refinement

To refine the pseudo-mask of the synthetic images, you need to modified the batch_infer.sh

Then, run the following command, after changing the log path in the **"batch_infer.sh"** file:
```bash
cd Polyp-PVT_box_guide
nohup bash bash/polyp/batch_infer.sh  > "your refinement log path" &

```


## Training Segmentation Model

### Preparations

Before running the training code, you need to download the pretrained model from [google drive](https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV), to the path of "Polyp-PVT/pretrained_pth".

### Training

To train the baseline segmentation model, you can run the following command, after changing the log path in the **"train.sh"** file:
```bash
cd Polyp-PVT
nohup bash bash/polyp/baseline/train.sh  > "your training log path" &
``` 

To train the augmentation segmentation model, you can run the following command, after changing the log path in the **"train_aug.sh"** file:
```bash
cd Polyp-PVT
nohup bash bash/polyp/aug/train_aug.sh  > "your training log path" &
``` 

By modified the "--align_score_cutoff" and "--prediction_score_cutoff" we can select different synthetic cases for model training.

## Tesing Segmentation Model

To test the segmentation model, you can run the following command, after changing the log path in the **"test.sh"** file:
```bash
cd Polyp-PVT
nohup bash bash/polyp/baseline/test.sh  > "your testing log path" &
```

```bash
cd Polyp-PVT
nohup bash bash/polyp/aug/test.sh  > "your testing log path" &
```

## 🤗 Acknowledgements
Codebase builds on [Diffusers](https://github.com/huggingface/diffusers) and [Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT).


<!--
**PolypInpainter** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->

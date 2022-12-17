<h1> Action Recognition for Surveillance </h1>

Using YOLOv5s to detect each person in the frame and use 
[AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) to get skeleton-pose and then use
[ST-GCN](https://github.com/yysijie/st-gcn) model to predict action from every 30 frames 
of each person tracks.

Which now support 7 actions: Standing, Walking, Sitting, Lying Down, Stand up, Sit down, Fall Down.

Using modfied YOLOv5 to detect helmet, cell phone, and smoking.
<div align="center">
    <img src="sample1.gif" width="416">
</div>

## Prerequisites

- Python >= 3.6
- torch >= 1.7.1
- opencv-python >= 4.1.2

You can also use "pip install -r requirements.txt" directly.

Original test run on: i7-8750H CPU @ 2.20GHz x12, GeForce RTX 2060 6GB, CUDA 10.2

## Data

Train with rotation augmented [COCO](http://cocodataset.org/#home) person keypoints dataset 
for more robust person detection in a variant of angle pose.

For actions recognition used data from [Le2i](http://le2i.cnrs.fr/Fall-detection-Dataset?lang=fr)
Fall detection Dataset (Coffee room, Home) extract skeleton-pose by AlphaPose and labeled each action 
frames by hand for training ST-GCN model.

## Pre-Trained Models

- SPPE FastPose (AlphaPose) - [resnet101](https://drive.google.com/file/d/1N2MgE1Esq6CKYA6FyZVKpPwHRyOCrzA0/view?usp=sharing),
[resnet50](https://drive.google.com/file/d/1IPfCDRwCmQDnQy94nT1V-_NVtTEi4VmU/view?usp=sharing)
- ST-GCN action recognition - [tsstg](https://drive.google.com/file/d/1mQQ4JHe58ylKbBqTjuKzpwN2nwKOWJ9u/view?usp=sharing)

## Configure environment variables

1. open ld.so.conf file

   ```bash
   sudo gedit /etc/ld.so.conf
   ```

2. add HikVision SDK lib path in new line

   ```bash
   Project_path/HiKcamSDK/lib
   ```

3. update .so library list

   ```
   sudo ldconfig
   ```

## Basic Use

1. Download all pre-trained models into ./Models folder.
2. Run my_multi_thread.py or Run image_input.py
```
    python my_multi_thread.py 
```

## Reference

- AlphaPose : https://github.com/Amanbhandula/AlphaPose
- ST-GCN : https://github.com/yysijie/st-gcn

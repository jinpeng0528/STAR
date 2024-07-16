# [NeurIPS 2023] Saving 100x Storage: Prototype Replay for Reconstructing Training Sample Distribution in Class-Incremental Semantic Segmentation

This is an official implementation of the paper "Saving 100x Storage: Prototype Replay for Reconstructing Training Sample Distribution in Class-Incremental Semantic Segmentation", accepted by NeurIPS 2023.
[[paper]](https://openreview.net/pdf?id=Ct0zPIe3xs)

## Installation
### Pre-requisites
This repository has been tested with the following environment:
* CUDA (11.3)
* Python (3.8.13)
* Pytorch (1.12.1)
* Pandas (2.0.3)

### Example conda environment setup
```bash
conda create -n star python=3.8.13
conda activate star
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install pandas==2.0.3
```

## Getting Started

### Datasets

#### PASCAL VOC 2012
We use augmented 10,582 training samples and 1,449 validation samples for PASCAL VOC 2012. You can download the original dataset in [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit). To train our model with augmented samples, please download labels of augmented samples (['SegmentationClassAug'](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip)) and file names (['train_aug.txt'](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/train_aug.txt)). The structure of data path should be organized as follows:
```bash
└── ./datasets/PascalVOC2012
    ├── Annotations
    ├── ImageSets
    │   └── Segmentation
    │       ├── train_aug.txt
    │       └── val.txt
    ├── JPEGImages
    ├── SegmentationClass
    ├── SegmentationClassAug
    └── SegmentationObject
    
    
```

#### ADE20K
We use 20,210 training samples and 2,000 validation samples for ADE20K. You can download the dataset in [here](http://sceneparsing.csail.mit.edu/). The structure of data path should be organized as follows:
```bash
└── ./datasets/ADE20K
    ├── annotations
    ├── images
    ├── objectInfo150.txt
    └── sceneCategories.txt
```

### Training
#### PASCAL VOC 2012
To train our model on the PASCAL VOC 2012 dataset, follow these example commands:
```Shell
GPU=0
BS=24
SAVEDIR='saved_voc'

TASKSETTING='disjoint'
TASKNAME='15-5'
INIT_LR=0.001
LR=0.0001
MEMORY_SIZE=0 # 50 for STAR-M

NAME='STAR'
python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}
```

#### ADE20K
To train our model on the PASCAL VOC 2012 dataset, follow these example commands:
```Shell
GPU=0,1
BS=12  # Total 24
SAVEDIR='saved_ade'

TASKSETTING='overlap'
TASKNAME='50-50'
INIT_LR=0.0025
LR=0.00025
MEMORY_SIZE=0

NAME='STAR'
python train_ade.py -c configs/config_ade.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS}

python train_ade.py -c configs/config_ade.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_ade.py -c configs/config_ade.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}
```

#### Scripts
To facilitate ease of use, you can directly run the `.sh` files located in the `./scripts/` directory. These files offer complete commands for training on both datasets.

### Testing
#### PASCAL VOC 2012
To evaluate on the PASCAL VOC 2012 dataset, execute the following command:
```Shell
python eval_voc.py --device 0 --test --resume path/to/weight.pth
```
Or, download our pretrained weights and corresponding `config.json` files provided below. Ensure that the config.json file is located in the same directory as the weight file.

| Method<br>(Overlapped) | 19-1<br>(2 steps) | 15-5<br>(2 steps) | 15-1<br>(6 steps) | 10-1<br>(11 steps) | 5-3<br>(6 steps) |
|:-----------------------|:-----------------:|:-----------------:|:-----------------:|:------------------:|:----------------:|
| STAR                   |     [76.61](https://drive.google.com/drive/folders/1MtuDShboaeNJ9gI_6m-XpGsoh9Nelj6W?usp=sharing)     |     [74.86](https://drive.google.com/drive/folders/1pSbZMQz8aQk9DkyhagQjclG8rO96vwOe?usp=sharing)     |     [72.90](https://drive.google.com/drive/folders/151x2QYEJp_rQRj9meA7xwY8GT7bOBlYE?usp=sharing)     |     [64.86](https://drive.google.com/drive/folders/1RgvPBHZUusjEasPHl6rGJ5aigLqA1Tvx?usp=sharing)      |    [64.54](https://drive.google.com/drive/folders/1GHpALBIegQXRqbM2b5E4ZJabfl27zVVp?usp=sharing)     |
| STAR-M                 |     [77.02](https://drive.google.com/drive/folders/1eoyag3QT64n3JfZkLjxHg5mx0LBkdmBi?usp=sharing)     |     [75.80](https://drive.google.com/drive/folders/15DWWHvIvB9ZGSdmEvdWbGKt68d_M8H0w?usp=sharing)     |     [74.03](https://drive.google.com/drive/folders/1iEV4p9-lhgIAZkbtyi2FjiZ1JMs1yQZ2?usp=sharing)     |     [66.60](https://drive.google.com/drive/folders/1oyJa_FKZ-8d1EOTb1pRWfeM-4jxuCwW3?usp=sharing)      |    [65.65](https://drive.google.com/drive/folders/1M91WkpH7nJLOf7_8bTLzDXtP2tkuwOnj?usp=sharing)     |

| Method<br>(Disjoint)  | 19-1<br>(2 steps) | 15-5<br>(2 steps) | 15-1<br>(6 steps) | 
|:----------------------|:-----------------:|:-----------------:|:-----------------:|
| STAR                  |     [76.38](https://drive.google.com/drive/folders/1E77dy7YhouEGkhJctEZ_5KWRNVpmuPxb?usp=sharing)     |     [73.48](https://drive.google.com/drive/folders/1k65FDEizrR4hnq5Bd0l8mQTHLs24S9t8?usp=sharing)     |     [70.77](https://drive.google.com/drive/folders/1vajiPbilmMR34NwwYYGIkmBqeX1OF_vg?usp=sharing)     |
| STAR-M                |     [76.73](https://drive.google.com/drive/folders/1BSZ8IoayV0obw33SyDxPoq0Hrbc86I3Z?usp=sharing)     |     [73.79](https://drive.google.com/drive/folders/1QgdxPXe7fQRoia-EgeyrAlf2O7zZoWY7?usp=sharing)     |     [71.18](https://drive.google.com/drive/folders/1vajiPbilmMR34NwwYYGIkmBqeX1OF_vg?usp=sharing)     |


#### ADE20K
To evaluate on the ADE20K dataset, execute the following command:
```Shell
python eval_ade.py --device 0 --test --resume path/to/weight.pth
```
Or, download our pretrained weights and corresponding `config.json` files provided below. Ensure that the config.json file is located in the same directory as the weight file.

| Method<br>(Disjoint)  | 100-50<br>(2 steps) | 100-10<br>(2 steps) | 50-50<br>(6 steps) | 
|:----------------------|:-------------------:|:-------------------:|:------------------:|
| STAR                  |      [36.39](https://drive.google.com/drive/folders/1nTz9cffAul-vnB3sCouinv1CAE9OnHN4?usp=sharing)      |      [34.91](https://drive.google.com/drive/folders/10bE9e1ms1C8AspeL45HA7hNHKSyfgDh5?usp=sharing)      |     [34.44](https://drive.google.com/drive/folders/1WjAduI5Q1CZMq0CwHLAHHVwswfXwlx8L?usp=sharing)      |


## Citation
```
@inproceedings{chen2023saving,
  title={Saving 100x Storage: Prototype Replay for Reconstructing Training Sample Distribution in Class-Incremental Semantic Segmentation},
  author={Chen, Jinpeng and Cong, Runmin and Luo, Yuxuan and Ip, Horace Ho Shing and Kwong, Sam},
  booktitle={NeurIPS},
  year={2023}
}
```

## Acknowledgements
* This code is based on [DKD](https://github.com/cvlab-yonsei/DKD) ([2022-NeurIPS] Decomposed Knowledge Distillation for Class-Incremental Semantic Segmentation).
* This template is borrowed from [pytorch-template](https://github.com/victoresque/pytorch-template).

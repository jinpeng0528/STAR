# [NeurIPS 2023] Saving 100x Storage: Prototype Replay for Reconstructing Training Sample Distribution in Class-Incremental Semantic Segmentation

This is an official implementation of the paper "Saving 100x Storage: Prototype Replay for Reconstructing Training Sample Distribution in Class-Incremental Semantic Segmentation", accepted by NeurIPS 2023.
📝 [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/708e0d691a22212e1e373dc8779cbe53-Abstract-Conference.html)
🤗 [Hugging Face](https://huggingface.co/jinpeng0528/STAR)

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
| STAR                   |     [76.61](https://huggingface.co/jinpeng0528/STAR/tree/main/voc_overlapped_19-1_STAR)     |     [74.86](https://huggingface.co/jinpeng0528/STAR/tree/main/voc_overlapped_15-5_STAR)     |     [72.90](https://huggingface.co/jinpeng0528/STAR/tree/main/voc_overlapped_15-1_STAR)     |     [64.86](https://huggingface.co/jinpeng0528/STAR/tree/main/voc_overlapped_10-1_STAR)      |    [64.54](https://huggingface.co/jinpeng0528/STAR/tree/main/voc_overlapped_5-3_STAR)     |
| STAR-M                 |     [77.02](https://huggingface.co/jinpeng0528/STAR/tree/main/voc_overlapped_19-1_STAR-M)     |     [75.80](https://huggingface.co/jinpeng0528/STAR/tree/main/voc_overlapped_15-5_STAR-M)     |     [74.03](https://huggingface.co/jinpeng0528/STAR/tree/main/voc_overlapped_15-1_STAR-M)     |     [66.60](https://huggingface.co/jinpeng0528/STAR/tree/main/voc_overlapped_10-1_STAR-M)      |    [65.65](https://huggingface.co/jinpeng0528/STAR/tree/main/voc_overlapped_5-3_STAR-M)     |

| Method<br>(Disjoint)  | 19-1<br>(2 steps) | 15-5<br>(2 steps) | 15-1<br>(6 steps) | 
|:----------------------|:-----------------:|:-----------------:|:-----------------:|
| STAR                  |     [76.38](https://huggingface.co/jinpeng0528/STAR/tree/main/voc_disjoint_19-1_STAR)     |     [73.48](https://huggingface.co/jinpeng0528/STAR/tree/main/voc_disjoint_15-5_STAR)     |     [70.77](https://huggingface.co/jinpeng0528/STAR/tree/main/voc_disjoint_15-1_STAR)     |
| STAR-M                |     [76.73](https://huggingface.co/jinpeng0528/STAR/tree/main/voc_disjoint_19-1_STAR-M)     |     [73.79](https://huggingface.co/jinpeng0528/STAR/tree/main/voc_disjoint_15-5_STAR-M)     |     [71.18](https://huggingface.co/jinpeng0528/STAR/tree/main/voc_disjoint_15-1_STAR-M)     |


#### ADE20K
To evaluate on the ADE20K dataset, execute the following command:
```Shell
python eval_ade.py --device 0 --test --resume path/to/weight.pth
```
Or, download our pretrained weights and corresponding `config.json` files provided below. Ensure that the config.json file is located in the same directory as the weight file.

| Method<br>(Disjoint)  | 100-50<br>(2 steps) | 100-10<br>(6 steps) | 50-50<br>(3 steps) | 
|:----------------------|:-------------------:|:-------------------:|:------------------:|
| STAR                  |      [36.39](https://huggingface.co/jinpeng0528/STAR/tree/main/ade_overlapped_100-50_STAR)      |      [34.91](https://huggingface.co/jinpeng0528/STAR/tree/main/ade_overlapped_100-10_STAR)      |     [34.44](https://huggingface.co/jinpeng0528/STAR/tree/main/ade_overlapped_50-50_STAR)      |


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

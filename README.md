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
| STAR                   |     [76.61](https://1drv.ms/f/c/7be8ecfc440137f7/EhIdDgqMcNRFsFe2bO3kwUoBqpfTil-nw8H8RfW-zyWBIg)     |     [74.86](https://1drv.ms/f/c/7be8ecfc440137f7/Eibeu5m4MJVKqtEBg8g0AJAB8NOGQdIS-UfouX8a6pWzHg)     |     [72.90](https://1drv.ms/f/c/7be8ecfc440137f7/Ermy2No82QhGi7s9wiS8ekgB3EyOsJCML60Zit6Qdasdwg)     |     [64.86](https://1drv.ms/f/c/7be8ecfc440137f7/EqvP86PEGeJHk2wV_qrdT3ABkV5ARlh3sPVR0DxxF4y2Ew)      |    [64.54](https://1drv.ms/f/c/7be8ecfc440137f7/El2E3MW7PDpHjbL69QIyj5MBPQrdZeGvI2PWnpaaubQ-TQ)     |
| STAR-M                 |     [77.02](https://1drv.ms/f/c/7be8ecfc440137f7/EnWyQVZya8tCrdpoeb0QQ2sBnCjsSaKApcoK1P5NtA4CAA)     |     [75.80](https://1drv.ms/f/c/7be8ecfc440137f7/Ep4M_3NNxFtCuIktk3rFIDoBbfKpj7rWsGIZ-OSjN335Ww)     |     [74.03](https://1drv.ms/f/c/7be8ecfc440137f7/EkWn2HverWxHpGpmp3lQJkgBQ2aCAvGO1CR-kNhdxpKxdA)     |     [66.60](https://1drv.ms/f/c/7be8ecfc440137f7/El-GKtKLDotAiBIECM9F2QQBxUQCusYdLC83bqseP5PnQA)      |    [65.65](https://1drv.ms/f/c/7be8ecfc440137f7/Eke6GbIC9UBPlNwRh6zLLWQB7aymtdtDGvgieZMMbSVWZw)     |

| Method<br>(Disjoint)  | 19-1<br>(2 steps) | 15-5<br>(2 steps) | 15-1<br>(6 steps) | 
|:----------------------|:-----------------:|:-----------------:|:-----------------:|
| STAR                  |     [76.38](https://1drv.ms/f/c/7be8ecfc440137f7/EjX7a2oZs-9Frv7BY1_U46wB9G3QSH5s-nn86YiLJ1qPew)     |     [73.48](https://1drv.ms/f/c/7be8ecfc440137f7/EqHNbJOZRwJDsoD3vcGiEe0BpEDHk-7b0D2aW2YulDKNZg)     |     [70.77](https://1drv.ms/f/c/7be8ecfc440137f7/Eksgoq5m1uRKn-zMWoRBJ4gBql_keybu4EbzceHh0sknEA)     |
| STAR-M                |     [76.73](https://1drv.ms/f/c/7be8ecfc440137f7/EoAsufpnOelBsl-2KSA2jnsBxIe-FLfBeiDjZzgQYf2ExA)     |     [73.79](https://1drv.ms/f/c/7be8ecfc440137f7/EgJXoK8XSUlCumiBERv49_gBgHOteDyJ8HClYLHrfJnoUA)     |     [71.18](https://1drv.ms/f/c/7be8ecfc440137f7/Esb-B8gjEdxOkVmDuR3wsmUBnpRj7jSzDLtboNmNO_Au3A)     |


#### ADE20K
To evaluate on the ADE20K dataset, execute the following command:
```Shell
python eval_ade.py --device 0 --test --resume path/to/weight.pth
```
Or, download our pretrained weights and corresponding `config.json` files provided below. Ensure that the config.json file is located in the same directory as the weight file.

| Method<br>(Disjoint)  | 100-50<br>(2 steps) | 100-10<br>(6 steps) | 50-50<br>(3 steps) | 
|:----------------------|:-------------------:|:-------------------:|:------------------:|
| STAR                  |      [36.39](https://1drv.ms/f/c/7be8ecfc440137f7/Ek4eU8D15jRMhLOWlxCYaA8BTtr2EqBC_Vv7LHIwmmERxA)      |      [34.91](https://1drv.ms/f/c/7be8ecfc440137f7/ElzKfcT511FOr5aconQh3BAB6kGsR5buK770cicmjP2Pnw)      |     [34.44](https://1drv.ms/f/c/7be8ecfc440137f7/ElRupd--D4JAvlHjFq1SdC0BcFdz3Dd0TjaN8BjTBEZxIQ)      |


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

<p align="center">
  <h1 align="center">Reason3D: Searching and Reasoning 3D Segmentation via Large Language Model [3DV 2025]
  </h1>
  <p align="center">
    <a href="https://kuanchihhuang.github.io/"><strong>Kuan-Chih Huang</strong></a>,
    <a href="https://lxtgh.github.io/"><strong>Xiangtai Li</strong></a>,
    <a href="https://luqi.info/"><strong>Lu Qi</strong></a>,
    <a href="https://yanshuicheng.info/"><strong>Shuicheng Yan</strong></a>,
    <a href="https://faculty.ucmerced.edu/mhyang/"><strong>Ming-Hsuan Yang</strong></a>
  </p>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2405.17427-red)](https://arxiv.org/abs/2405.17427)
[![Project](https://img.shields.io/badge/project-page-green)](https://kuanchihhuang.github.io/project/reason3d/)

</div>

This branch contains the hierarchical searching code of our Reason3D model.

## Overview

<img src="figs/reason3d_arch.jpg" alt="vis" style="zoom:50%;" />

We introduce Reason3D, a novel LLM for comprehensive 3D understanding that processes point cloud data and text prompts to produce textual responses and segmentation masks. This enables advanced tasks such as 3D reasoning segmentation, hierarchical searching, referring expressions, and question answering with detailed mask outputs.

## Installation

1. Create conda environment. We use `python=3.8` `pytorch=1.11.0` and `cuda=11.3`.
```bash
conda create -n reason3d python=3.8
conda activate reason3d
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

2. Install [LAVIS](https://github.com/salesforce/LAVIS)
```bash
git clone https://github.com/salesforce/LAVIS.git SalesForce-LAVIS
cd SalesForce-LAVIS
pip install -e .
```

3. Install segmentor from this [repo](https://github.com/Karbo123/segmentator) (used for superpoint construction). We also provide an alternative PyTorch implementation `segmentator_pytorch.py`, though it may yield slightly lower performance.


4. Install pointgroup_ops
```bash
cd lavis/models/reason3d_models/lib
sudo apt-get install libsparsehash-dev
python setup.py develop
```

## Data Preparation

### Matterport3D dataset

Please follow the instructions [here](https://niessner.github.io/Matterport/) to access official `download_mp.py` script, run the following in `data/matterport/`:
```
python2 download_mp.py -o . --type region_segmentations
```
Extract files and organize data as follows:
```
Reason3D
├── data
│   ├── matterport
│   │   ├── scans
│   │   │   ├── 17DRP5sb8fy
│   │   │   │   ├──region_segmentations
│   │   │   │   │   ├──region0.ply
│   │   │   ├── ...
```
Process data on Matterport3D dataset for 3D reasoning segmentation task:
```
cd data/matterport
python3 process_mp3d.py
```
After running the script, the Matterport3D dataset structure should look like below.
```
Reason3D
├── data
│   ├── matterport
│   │   ├── mp3d_data
│   │   │   ├── XXXXX_regionX.pth
│   │   │   ├── ...
```

You can directly download our preprocessed data ([mp3d_data.zip](https://drive.google.com/file/d/1OXT_hmv-9eHgqpcl3A0V28y-DfC5v0-y/view)), please agree the official license before download it.

For our searching task, we need to get the type of each room for Matterport3D dataset, run the following in `data/matterport/`:
```
python2 download_mp.py -o . --type house_segmentations
```
We only require `xxxxx.house` files. Extract files and organize data as follows:
```
Reason3D
├── data
│   ├── matterport
│   │   ├── house_type
│   │   │   ├── 17DRP5sb8fy.house
│   │   │   ├── 1LXtFkjw3qL.house
│   │   │   ├── ...
```
Extract the room type information:
```
cd data/matterport
python3 parse_mp3d_house_type.py
```
After that, we will get `mp3d_room_type.json` under `data/matterport`. You can directly download our file [here](https://drive.google.com/file/d/1ZSFgZbpGLn79Ih_XJhNhWQJ9rTRtLn_e/view?usp=sharing) to skip this step.

### Reason3D dataset (Searching)

Download sample searching annotations to quickly set up and test our pipeline from this [link](https://drive.google.com/file/d/1Fjv0G6zCLLt0QvbASkP5eTiwWR9gwwv0/view?usp=sharing).

After downloading, place the files in the following directory structure:

```
Reason3D
├── data
│   ├── reason3d_search
│   │   ├── mp3d_room_train.json
│   │   ├── mp3d_room_val.json
```

Note: This dataset is provided only as sample data to demonstrate the functionality and workflow of our coarse-to-fine pipeline.

## Pretrained Backbone
Download the [SPFormer](https://github.com/sunjiahao1999/SPFormer) pretrained backbone (or provided by [3D-STMN](https://github.com/sosppxo/3D-STMN)) and move it to checkpoints.
```
mkdir checkpoints
mv ${Download_PATH}/sp_unet_backbone.pth checkpoints/
```
You can also pretrain the backbone by yourself and modify the path [here](lavis/projects/reason3d/train/reason3d_scanrefer_scratch.yaml#L15).

## Training
- **Pretraining:** Pretrain on ScanRefer dataset:
```
python -m torch.distributed.run --nproc_per_node=4 --master_port=29501 train.py --cfg-path lavis/projects/reason3d/train/reason3d_scanrefer_scratch.yaml
```
Note: we set a certain range around the target object to represent the coarse target region.

- **3D reasoning segmentation (searching):** Train on Reason3D dataset (searching) using the pretrained checkpoint from the 3D referring segmentation model:
```
python -m torch.distributed.run --nproc_per_node=2 --master_port=29501 train.py --cfg-path lavis/projects/reason3d/train/reason3d_reason.yaml --options model.pretrained=<path_to_pretrained_checkpoint>
```
Replace `<path_to_pretrained_checkpoint>` with the path to your pretrained model. For example: `./lavis/output/reason3d/xxxx/checkpoint_xx.pth`


## Evaluation

- **3D reasoning segmentation (searching):** Evaluate on our Reason3D dataset: 
```
python evaluate.py --cfg-path lavis/projects/reason3d/val/reason3d_reason.yaml --options model.pretrained=<path_to_pretrained_checkpoint> run.save_results=True
```
Add `run.save_results=True` option if you want to save prediction results.

Note: we use a room number of 3 for both training and testing for demonstration purposes.

## Acknowlegment

Our codes are mainly based on [LAVIS](https://github.com/salesforce/LAVIS), [3D-LLM](https://github.com/UMass-Foundation-Model/3D-LLM) and [3D-STMN](https://github.com/sosppxo/3D-STMN). Thanks for their contributions!


## Citation

If you find our work useful for your project, please consider citing our paper:


```bibtex
@article{reason3d,
  title={Reason3D: Searching and Reasoning 3D Segmentation via Large Language Model},
  author={Kuan-Chih Huang and Xiangtai Li and Lu Qi and Shuicheng Yan and Ming-Hsuan Yang},
  journal={3DV},
  year={2025}
}
```

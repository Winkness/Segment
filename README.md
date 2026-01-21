## DSAM-Seg: Towards Limited Labeled Data and Cross-Region Road Extraction by Parameter-Efficient Transfer Learning
![framework](./fig1_01.png)
## News
-2026/1/21-30 Code is being organized and uploaded.
## Requirements
-Pytorch 2.7.1
## Clone Repository
```shell
git clone https://github.com/Winkness/Segment.git
cd Segment
```
## Get Started Quickly
```shell
conda create -n DSAM-Seg python==3.10
conda activate DSAM-Seg
pip install -r requirements.txt
```
## Datasets
- [DeepGlobe](https://competitions.codalab.org/competitions/18467#participate-get_starting_kit)
- [Massachusetts](https://www.cs.toronto.edu/~vmnih/data/)
- [SpaceNet-Test, From WHU](https://rsidea.whu.edu.cn/GRSet.htm)
- [RoadTracer-Test, From WHU](https://rsidea.whu.edu.cn/GRSet.htm)
## Preprocessing Datasets
```shell
cd Process
python split.py
```
## Train
```shell
cd Segment

CUDA_VISIBLE_DEVICES="1" \
python train.py --hiera_path "/insert your path/pretrained-SAM/sam2_hiera_large.pt" \
--dinov3_path "./insert your path/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth" \
--train_image_path "/insert your train dataset path/images/" \
--train_mask_path "/insert your train dataset path/masks/" \
--save_path "./save" \
--epoch 20 \
--lr 0.0002 \
--batch_size 2
```
## Test
```shell
CUDA_VISIBLE_DEVICES="1" \
python test.py \
--checkpoint "/insert your model path/DSAN-Seg-20.pth" \
--test_image_path "/insert your test dataset path/img/" \
--test_gt_path "/insert your test dataset path/gt/" \
--save_path "./masks"
```
## Acknowledgement
-Pretrained Model
[segment anything 2](https://github.com/facebookresearch/segment-anything-2)
[DINOV3](https://github.com/facebookresearch/dinov3.git)
-Datasets
[The Deepglobe dataset](https://competitions.codalab.org/competitions/18467#participate-get_starting_kit)
[The Massachusetts dataset](https://www.cs.toronto.edu/~vmnih/data/)
[The GRSet dataset](https://rsidea.whu.edu.cn/GRSet.htm)

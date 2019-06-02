# Semi-supervised Transfer Learning for Image Rain Removal

This package contains the Python implementation of "Semi-supervised Transfer Learning for Image Rain Removal", in CVPR 2019.





## Usage

#### Prepare Training Data
Download synthesized data from [here](https://pan.baidu.com/s/1Hvm9ctniC7PMQdKrI_lf3Q), as supervised training data. Put input images in './data/rainy_image_dataset/input' and ground truth images in './data/rainy_image_dataset/label'.
Run /data/generate.m to generate HDF files as training data.

#### Train
python train.py

#### Test
python test.py

## Cite
Please cite this paper if you use the code:
  Wei Wei, Deyu Meng, Qian Zhao, Zongben Xu and Ying Wu, "Semi-supervised Transfer Learning for Image Rain Removal", in IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR'19), Long Beach, CA, June, 2019.

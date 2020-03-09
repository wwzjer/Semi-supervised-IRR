# Semi-supervised Transfer Learning for Image Rain Removal

This package contains the Python implementation of "[Semi-supervised Transfer Learning for Image Rain Removal](https://arxiv.org/pdf/1807.11078.pdf)", in CVPR 2019.





## Usage

#### Prepare Training Data
Download synthesized data from [here](https://github.com/jinnovation/rainy-image-dataset), as supervised training data. Put input images in './data/rainy_image_dataset/input' and ground truth images in './data/rainy_image_dataset/label'.
Run /data/generate.m to generate HDF files as training data.

#### Train
python training.py

#### Test
python testing.py

## Cite
Please cite this paper if you use the code:

    @InProceedings{Wei_2019_CVPR,
    author = {Wei, Wei and Meng, Deyu and Zhao, Qian and Xu, Zongben and Wu, Ying},
    title = {Semi-Supervised Transfer Learning for Image Rain Removal},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition},
    year = {2019}
    }
    
The prior work using MoG for video deraining can be found here:

    @InProceedings{wei2017should,
      title={Should we encode rain streaks in video as deterministic or stochastic?},
      author={Wei, Wei and Yi, Lixuan and Xie, Qi and Zhao, Qian and Meng, Deyu and Xu, Zongben},
      booktitle={Proceedings of the IEEE International Conference on Computer Vision},
      year={2017}
    }

## Acknowledge
We use [Deep Detail Network](https://xueyangfu.github.io/projects/cvpr2017.html) as our baseline. Thanks for sharing the code!

## Note
1. You are welcomed to add more real data.
2. You are welcomed to try more recent derain network as baseline.

# DCTON (CVPR 2021)

**Disentangled Cycle Consistency for Highly-realistic Virtual Try-On (https://arxiv.org/abs/2103.09479)**

![image](https://github.com/ChongjianGE/DCTON/blob/main/image/show.png?raw=true)

## Prerequisites
- python 3.6
- pytorch 1.0.0
- torchvision 0.3.0
- cuda 10.0
- opencv

To install requirements:
```setup
conda create -n dcton python=3.6
conda activate dcton
conda install pytorch==1.0.10 torchvision==0.3.0 cuda100
pip install tensorboardX
pip install opencv-python
pip install imdb
pip install tqdm
```

## Dataset
For data preparation, please refer to [VITON](https://github.com/xthan/VITON).

## Run the Demo
Download [trained weights](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/rhettgee_connect_hku_hk/Es8E9XizOndEmTcG9KPiLJIBIxv0Gke9RlPPxQbGmHbOVA?e=MTLgoG).
Put the trained weights in the 'pretrained_model' file.

We here provide some data in 'demo_data' file for demo running.
```setup
# Demo data running
bash test.sh
```

## License
The use of this code is restricted to non-commercial research.

## Acknowledgement 
Thanks for [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) for providing the useful codes.

## Citation
If you think our work is useful, please feel free to cite.
```
@inproceedings{ge2021disentangled,
  title={Disentangled Cycle Consistency for Highly-realistic Virtual Try-On},
  author={Ge, Chongjian and Song, Yibing and Ge, Yuying and Yang, Han and Liu, Wei and Luo, Ping},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16928--16937},
  year={2021}
}
```




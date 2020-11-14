# CPASSRnet-Tensorflow
Simple Tensorflow implementation of "Cross Parallax Attention Network for Stereo Super-resolution"

## Requirements
* Tensorflow 1.9
* Python 3.6
* numpy
* scipy
* scikit-image
* pillow
* pandas
* vgg19 pretrained model, download URL : http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat

## Usage
```
├── datasets
   └── YOUR_DATASET_NAME
       ├── train_patches_96_288 (patch size 96x288)
           ├── xxx_L.jpg (format doesn't matter, but 'name_L','name_R' must be corresponding. e.g. patch_1_L.png and patch_1_R.png)
           ├── xxx_R.png
           ├── yyy_L.jpg
           └── ...
       ├── test
           ├── aaa_L.jpg 
           ├── aaa_R.png 
           ├── bbb_L.png
           └── ...
├── pretrained_vgg
   └── imagenet-vgg-verydeep-19.mat
```

### Train
* python main.py --phase=train --dataset=middlebury --batch_size=6 --visible_devices=0

### Test
* python main.py --phase=test --dataset=middlebury --visible_devices=0


## Related works
* [PASSRnet-Pytorch](https://github.com/LongguangWang/PASSRnet)
* [StereoSR-Tensorflow](https://github.com/PeterZhouSZ/stereosr)

## Reference & Thanks
* [MUNIT-Tensorflow](https://github.com/taki0112/MUNIT-Tensorflow)

## Author
Canqiang Chen

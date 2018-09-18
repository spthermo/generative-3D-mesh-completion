# Generative 3D Mesh Completion
This work extends the potential of the GANs to the 3D domain. The presented 3D GAN (generator-discriminator) receives corrupted 3D meshes of [ShapeNet](https://www.shapenet.org/) and generates "denoised" samples. The 3D meshes are corrupted at random positions with "cuboid" noise. Some indicative results are under the "examples directory".

## Prerequisites
The architecture has been implemented using the following:
- Python 3.5
- Scipy, Scikit-image
- Torchvision
- Tensorflow 1.7.0
- Tensorboard
- visdom (follow the steps [here](https://github.com/facebookresearch/visdom))

Tensorflow and Tensorboard are used for visualization and monitoring purposes, thus they are not mandatory.

## Model architecture details
Generator details:

|      Type      | Kernel | Dilation | Stride | Output |
|:--------------:|:------:|:--------:|:------:|:------:|
|     conv3d     |  4x4   |    1     |  2x2   |   64   |
|     conv3d     |  4x4   |    1     |  2x2   |   128  |
|     conv3d     |  4x4   |    1     |  2x2   |   128  |
|     conv3d     |  3x3   |    1     |  1x1   |   256  |
| dilated conv3d |  3x3   |    4     |  1x1   |   256  |
| dilated conv3d |  3x3   |    8     |  1x1   |   256  |
|     conv3d     |  3x3   |    1     |  1x1   |   256  |
|    deconv3d    |  4x4   |    1     |  2x2   |   128  |
|    deconv3d    |  4x4   |    1     |  2x2   |   64   |
|    deconv3d    |  4x4   |    1     |  2x2   |   1    |

Discriminator details:

|      Type      | Kernel | Dilation | Stride | Output |
|:--------------:|:------:|:--------:|:------:|:------:|
|     conv3d     |  4x4   |    1     |  2x2   |   64   |
|     conv3d     |  4x4   |    1     |  2x2   |   128  |
|     conv3d     |  4x4   |    1     |  2x2   |   256  |
|     conv3d     |  4x4   |    1     |  2x2   |   512  |



## Training with ShapeNet
The generator (completion netwrok) is initially warmed up for 20 epochs. After that, both the generator and disriminator are trained for 80 epochs. Train using:

```
python train.py
```

Edit the path to 3DShapeNet in ```dataIO.py``` script. This script is responsible for loading ".off" files and transform them to voxels (numpy). The ```logger.py``` file is used to create and update the model's instance for Tensorboard. To monitor the training process use:

```
tensorboard --logdir='./logs' --port 6006
```
and use your browser to access the localhost at the specified port.


## 3D mesh completion examples
Some indicative 3DShapeNet samples (using "chair" meshes) in "real-noisy-generated" triplet form:

<img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/1_1.png" width="200"> <img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/1_2.png" width="200"> <img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/1_3.png" width="200">

<img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/1_4.png" width="200"> <img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/1_5.png" width="200"> <img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/1_6.png" width="200">

___


<img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/2_1.png" width="200"> <img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/2_2.png" width="200"> <img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/2_3.png" width="200">

<img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/2_4.png" width="200"> <img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/2_5.png" width="200"> <img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/2_6.png" width="200">

___


<img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/3_1.png" width="200"> <img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/3_2.png" width="200"> <img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/3_3.png" width="200">

<img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/3_4.png" width="200"> <img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/3_5.png" width="200"> <img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/3_6.png" width="200">

___


More challenging examples:

<img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/4_1.png" width="200"> <img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/4_2.png" width="200"> <img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/4_3.png" width="200">

<img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/4_4.png" width="200"> <img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/4_5.png" width="200"> <img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/4_6.png" width="200">

___


<img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/5_1.png" width="200"> <img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/5_2.png" width="200"> <img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/5_3.png" width="200">

<img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/5_4.png" width="200"> <img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/5_5.png" width="200"> <img src="https://github.com/spthermo/generative-3D-mesh-completion/blob/master/examples/5_6.png" width="200">

## To do
Add a local discriminator and test it on unseen 3D meshes.


## Acknowledgement
The Tensorboard support is provided from [yunjey](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard)
[Visdom]((https://github.com/facebookresearch/visdom)) visualization tool - Facebook research


### Similar projects:
[tf-3dgan](https://github.com/meetshah1995/tf-3dgan) is a similar approach developed in TensorFlow.

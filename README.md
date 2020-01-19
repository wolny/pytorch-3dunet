[![DOI](https://zenodo.org/badge/149826542.svg)](https://zenodo.org/badge/latestdoi/149826542)

# pytorch-3dunet

PyTorch implementation 3D U-Net and its variants:

- Standard 3D U-Net based on [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/abs/1606.06650) 
Özgün Çiçek et al.

- Residual 3D U-Net based on [Superhuman Accuracy on the SNEMI3D Connectomics Challenge](https://arxiv.org/pdf/1706.00120.pdf) Kisuk Lee et al.

## Prerequisites
- Linux
- NVIDIA GPU
- CUDA CuDNN

## Supported model architectures
- in order to train standard 3D U-Net specify `name: UNet3D` in the `model` section of the [config file](resources/train_config_ce.yaml)
- in order to train Residual U-Net specify `name: ResidualUNet3D` in the `model` section of the [config file](resources/train_config_ce.yaml)

## Supported Loss Functions
For a detailed explanation of the loss functions used see:
[Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations](https://arxiv.org/pdf/1707.03237.pdf)
Carole H. Sudre, Wenqi Li, Tom Vercauteren, Sebastien Ourselin, M. Jorge Cardoso

- _WeightedCrossEntropyLoss_ (see 'Weighted cross-entropy (WCE)' in the above paper for a detailed explanation; one can specify class weights via `weight: [w_1, ..., w_k]` in the `loss` section of the config)
- _CrossEntropyLoss_ (one can specify class weights via `weight: [w_1, ..., w_k]` in the `loss` section of the config)
- _PixelWiseCrossEntropyLoss_ (one can specify not only class weights but also per pixel weights in order to give more gradient to important (or under-represented) regions in the ground truth)
- _BCEWithLogitsLoss_ (binary cross-entropy)
- _DiceLoss_ standard Dice loss (standard `DiceLoss` defined as `1 - DiceCoefficient` used for binary semantic segmentation; when more than 2 classes are present in the ground truth, it computes the `DiceLoss` per channel and averages the values).
- _GeneralizedDiceLoss_ (see 'Generalized Dice Loss (GDL)' in the above paper for a detailed explanation; one can specify class weights via `weight: [w_1, ..., w_k]` in the `loss` section of the config). 
Note: use this loss function only if the labels in the training dataset are very imbalanced e.g. one class having at least 3 orders of magnitude more voxels than the others. Otherwise use standard _DiceLoss_. 


## Supported Evaluation Metrics
Standard semantic segmentation metrics:
- **MeanIoU** - Mean intersection over union
- **DiceCoefficient** - Dice Coefficient (computes per channel Dice Coefficient and returns the average)
If a 3D U-Net was trained to predict cell boundaries, one can use the following semantic instance segmentation metrics
(the metrics below are computed by running connected components on thresholded boundary map and comparing the resulted instances to the ground truth instance segmentation): 
- **BoundaryAveragePrecision** - Average Precision
- **AdaptedRandError** - Adapted Rand Error (see http://brainiac2.mit.edu/SNEMI3D/evaluation for a detailed explanation)

If not specified `MeanIoU` will be used by default.


## Getting Started

## Installation
- The easiest way to install `pytorch-3dunet` package is via conda:
```
conda create --name 3dunet python=3.7
conda activate 3dunet 
conda install -c conda-forge -c awolny pytorch-3dunet
```
After installation the following commands are accessible within the conda environment:
`train3dunet` for training the network and `predict3dunet` for prediction (see below).

- One can also install directly from source:
```
python setup.py install
```

## Train
Given that `pytorch-3dunet` package was installed via conda as described above, one can train the network by simply invoking:
```
train3dunet --config <CONFIG>
```
where `CONFIG` is the path to a YAML configuration file, which specifies all aspects of the training procedure.

See e.g. [train_config_ce.yaml](resources/train_config_ce.yaml) which describes how to train a standard 3D U-Net on a randomly generated 3D volume and random segmentation mask ([random_label3D.h5](resources/random_label3D.h5)) with cross-entropy loss (just a demo). 

In order to train on your own data just provide the paths to your HDF5 training and validation datasets in the [train_config_ce.yaml](resources/train_config_ce.yaml).
The HDF5 files should contain the raw/label data sets in the following axis order: `DHW` (in case of 3D) `CDHW` (in case of 4D).

One can monitor the training progress with Tensorboard `tensorboard --logdir <checkpoint_dir>/logs/` (you need `tensorflow` installed in your conda env), where `checkpoint_dir` is the path to the checkpoint directory specified in the config.

To try out training on randomly generated data right away:
1. Checkout the repository
2. 
```
cd pytorch3dunet
train3dunet --config ../resources/train_config_ce.yaml # train with CrossEntropyLoss
train3dunet --config ../resources/train_config_dice.yaml # train with DiceLoss
```


### Training tips
When training with binary-based losses, i.e.: `BCEWithLogitsLoss`, `DiceLoss`, `GeneralizedDiceLoss`:
1. the label data has to be 4D (one target binary mask per channel).
If you have a 3D binary data (foreground/background), you can just change `ToTensor` transform for the label to contain `expand_dims: true`, see e.g. [train_config_dice.yaml](resources/train_config_dice.yaml).
2. `final_sigmoid=True` has to be present in the `model` section of the config, since every output channel gives the probability of the foreground.
When training with cross entropy based losses (`WeightedCrossEntropyLoss`, `CrossEntropyLoss`, `PixelWiseCrossEntropyLoss`) set `final_sigmoid=False` so that `Softmax` normalization is applied to the output.

## Test
Given that `pytorch-3dunet` package was installed via conda as described above, one can run the prediction via:
```
predict3dunet --config <CONFIG>
```

To run the prediction on randomly generated 3D volume (just for demonstration purposes) from [random_label3D.h5](resources/random_label3D.h5) and a network trained with cross-entropy loss:
```
cd pytorch3dunet
predict3dunet --config ../resources/test_config_ce.yaml

```
or if trained with `DiceLoss`:
```
cd pytorch3dunet
predict3dunet --config ../resources/test_config_dice.yaml

```
Predicted volume will be saved to `resources/random_label3D_probabilities.h5`.

In order to predict on your own data, just provide the path to your model as well as paths to HDF5 test files (see[test_config_ce.yaml](resources/test_config_ce.yaml)).

### Prediction tips
In order to avoid checkerboard artifacts in the output prediction masks the patch predictions are averaged, so make sure that `patch/stride` params lead to overlapping blocks, e.g. `patch: [64 128 128] stride: [32 96 96]` will give you a 'halo' of 32 voxels in each direction.

## Contribute
If you want to contribute back, please make a pull request.

## Cite
If you use this code for your research, please cite as:

Adrian Wolny. (2019, May 7). wolny/pytorch-3dunet: PyTorch implementation of 3D U-Net (Version v1.0.0). Zenodo. http://doi.org/10.5281/zenodo.2671581




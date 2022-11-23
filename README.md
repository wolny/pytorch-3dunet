[![DOI](https://zenodo.org/badge/149826542.svg)](https://doi.org/10.7554/eLife.57613)
[![Build Status](https://github.com/wolny/pytorch-3dunet/actions/workflows/conda-build.yml/badge.svg)](https://github.com/wolny/pytorch-3dunet/actions/)
[![Anaconda-Server Badge](https://anaconda.org/awolny/pytorch-3dunet/badges/latest_release_date.svg)](https://anaconda.org/awolny/pytorch-3dunet)
[![Anaconda-Server Badge](https://anaconda.org/awolny/pytorch-3dunet/badges/downloads.svg)](https://anaconda.org/awolny/pytorch-3dunet)
[![Anaconda-Server Badge](https://anaconda.org/awolny/pytorch-3dunet/badges/license.svg)](https://anaconda.org/awolny/pytorch-3dunet)

# pytorch-3dunet

PyTorch implementation 3D U-Net and its variants:

- Standard 3D U-Net based on [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/abs/1606.06650) 
Özgün Çiçek et al.

- Residual 3D U-Net based on [Superhuman Accuracy on the SNEMI3D Connectomics Challenge](https://arxiv.org/pdf/1706.00120.pdf) Kisuk Lee et al.

The code allows for training the U-Net for both: **semantic segmentation** (binary and multi-class) and **regression** problems (e.g. de-noising, learning deconvolutions).

## 2D U-Net
Training the standard 2D U-Net is also possible, see [2DUnet_dsb2018](resources/2DUnet_dsb2018/train_config.yml) for example configuration. Just make sure to keep the singleton z-dimension in your H5 dataset (i.e. `(1, Y, X)` instead of `(Y, X)`) , because data loading / data augmentation requires tensors of rank 3 always.

## Prerequisites
- Linux
- NVIDIA GPU
- CUDA CuDNN

### Running on Windows
The package has not been tested on Windows, however some users reported using it successfully on Windows.


## Supported Loss Functions

### Semantic Segmentation
- _BCEWithLogitsLoss_ (binary cross-entropy)
- _DiceLoss_ (standard `DiceLoss` defined as `1 - DiceCoefficient` used for binary semantic segmentation; when more than 2 classes are present in the ground truth, it computes the `DiceLoss` per channel and averages the values)
- _BCEDiceLoss_ (Linear combination of BCE and Dice losses, i.e. `alpha * BCE + beta * Dice`, `alpha, beta` can be specified in the `loss` section of the config)
- _CrossEntropyLoss_ (one can specify class weights via the `weight: [w_1, ..., w_k]` in the `loss` section of the config)
- _PixelWiseCrossEntropyLoss_ (one can specify per pixel weights in order to give more gradient to the important/under-represented regions in the ground truth)
- _WeightedCrossEntropyLoss_ (see 'Weighted cross-entropy (WCE)' in the below paper for a detailed explanation)
- _GeneralizedDiceLoss_ (see 'Generalized Dice Loss (GDL)' in the below paper for a detailed explanation) Note: use this loss function only if the labels in the training dataset are very imbalanced e.g. one class having at least 3 orders of magnitude more voxels than the others. Otherwise use standard _DiceLoss_.

For a detailed explanation of some of the supported loss functions see:
[Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations](https://arxiv.org/pdf/1707.03237.pdf)
Carole H. Sudre et al.

### Regression
- _MSELoss_ (mean squared error loss)
- _L1Loss_ (mean absolute errro loss)
- _SmoothL1Loss_ (less sensitive to outliers than MSELoss)
- _WeightedSmoothL1Loss_ (extension of the _SmoothL1Loss_ which allows to weight the voxel values above/below a given threshold differently)


## Supported Evaluation Metrics

### Semantic Segmentation
- _MeanIoU_ (mean intersection over union)
- _DiceCoefficient_ (computes per channel Dice Coefficient and returns the average)
If a 3D U-Net was trained to predict cell boundaries, one can use the following semantic instance segmentation metrics
(the metrics below are computed by running connected components on thresholded boundary map and comparing the resulted instances to the ground truth instance segmentation): 
- _BoundaryAveragePrecision_ (Average Precision applied to the boundary probability maps: thresholds the output from the network, runs connected components to get the segmentation and computes AP between the resulting segmentation and the ground truth)
- _AdaptedRandError_ (see http://brainiac2.mit.edu/SNEMI3D/evaluation for a detailed explanation)
- _AveragePrecision_ (see https://www.kaggle.com/stkbailey/step-by-step-explanation-of-scoring-metric)

If not specified `MeanIoU` will be used by default.


### Regression
- _PSNR_ (peak signal to noise ratio)
- _MSE_ (mean squared error)


## Installation
- The easiest way to install `pytorch-3dunet` package is via conda:
```
conda create -n pytorch3dunet -c pytorch -c conda-forge -c awolny pytorch-3dunet
conda activate pytorch3dunet
```
After installation the following commands are accessible within the conda environment:
`train3dunet` for training the network and `predict3dunet` for prediction (see below).

- One can also install directly from source:
```
python setup.py install
```

### Installation tips
Make sure that the installed `pytorch` is compatible with your CUDA version, otherwise the training/prediction will fail to run on GPU. You can re-install `pytorch` compatible with your CUDA in the `pytorch3dunet` environment by:
```
conda install -c pytorch cudatoolkit=<YOU_CUDA_VERSION> pytorch
```

## Train
Given that `pytorch-3dunet` package was installed via conda as described above, one can train the network by simply invoking:
```
train3dunet --config <CONFIG>
```
where `CONFIG` is the path to a YAML configuration file, which specifies all aspects of the training procedure. 

In order to train on your own data just provide the paths to your HDF5 training and validation datasets in the config.

* sample config for 3D semantic segmentation (cell boundary segmentation): [train_config_segmentation.yaml](resources/3DUnet_confocal_boundary/train_config.yml))
* sample config for 3D regression task (denoising): [train_config_regression.yaml](resources/3DUnet_denoising/train_config_regression.yaml))

The HDF5 files should contain the raw/label data sets in the following axis order: `DHW` (in case of 3D) `CDHW` (in case of 4D).

One can monitor the training progress with Tensorboard `tensorboard --logdir <checkpoint_dir>/logs/` (you need `tensorflow` installed in your conda env), where `checkpoint_dir` is the path to the checkpoint directory specified in the config.

### Training tips
1. When training with binary-based losses, i.e.: `BCEWithLogitsLoss`, `DiceLoss`, `BCEDiceLoss`, `GeneralizedDiceLoss`:
The target data has to be 4D (one target binary mask per channel).
When training with `WeightedCrossEntropyLoss`, `CrossEntropyLoss`, `PixelWiseCrossEntropyLoss` the target dataset has to be 3D, see also pytorch documentation for CE loss: https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html
2. `final_sigmoid` in the `model` config section applies only to the inference time (validation, test):
When training with cross entropy based losses (`WeightedCrossEntropyLoss`, `CrossEntropyLoss`, `PixelWiseCrossEntropyLoss`) set `final_sigmoid=False` so that `Softmax` normalization is applied to the output.
When training with `BCEWithLogitsLoss`, `DiceLoss`, `BCEDiceLoss`, `GeneralizedDiceLoss` set `final_sigmoid=True`

## Prediction
Given that `pytorch-3dunet` package was installed via conda as described above, one can run the prediction via:
```
predict3dunet --config <CONFIG>
```

In order to predict on your own data, just provide the path to your model as well as paths to HDF5 test files (see example [test_config_segmentation.yaml](resources/3DUnet_confocal_boundary/test_config.yml)).

### Prediction tips
In order to avoid patch boundary artifacts in the output prediction masks the patch predictions are averaged, so make sure that `patch/stride` params lead to overlapping blocks, e.g. `patch: [64, 128, 128] stride: [32, 96, 96]` will give you a 'halo' of 32 voxels in each direction.

## Data Parallelism
By default, if multiple GPUs are available training/prediction will be run on all the GPUs using [DataParallel](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html).
If training/prediction on all available GPUs is not desirable, restrict the number of GPUs using `CUDA_VISIBLE_DEVICES`, e.g.
```bash
CUDA_VISIBLE_DEVICES=0,1 train3dunet --config <CONFIG>
``` 
or
```bash
CUDA_VISIBLE_DEVICES=0,1 predict3dunet --config <CONFIG>
```

## Examples

### Cell boundary predictions for lightsheet images of Arabidopsis thaliana lateral root
Training/predictions configs can be found in [3DUnet_lightsheet_boundary](resources/3DUnet_lightsheet_boundary).
Pre-trained model weights available [here](https://oc.embl.de/index.php/s/61s67Mg5VQy7dh9/download?path=%2FLateral-Root-Primordia%2Funet_bce_dice_ds1x&files=best_checkpoint.pytorch).
In order to use the pre-trained model on your own data:
* download the `best_checkpoint.pytorch` from the above link
* add the path to the downloaded model and the path to your data in [test_config.yml](resources/3DUnet_lightsheet_boundary/test_config.yml)
* run `predict3dunet --config test_config.yml`
* optionally fine-tune the pre-trained model with your own data, by setting the `pre_trained` attribute in the YAML config to point to the `best_checkpoint.pytorch` path

The data used for training can be downloaded from the following OSF project:
* training set: https://osf.io/9x3g2/
* validation set: https://osf.io/vs6gb/
* test set: https://osf.io/tn4xj/

Sample z-slice predictions on the test set (top: raw input , bottom: boundary predictions):

<img src="https://github.com/wolny/pytorch-3dunet/blob/master/resources/3DUnet_lightsheet_boundary/root_movie1_t45_raw.png" width="400">
<img src="https://github.com/wolny/pytorch-3dunet/blob/master/resources/3DUnet_lightsheet_boundary/root_movie1_t45_pred.png" width="400">

### Cell boundary predictions for confocal images of Arabidopsis thaliana ovules
Training/predictions configs can be found in [3DUnet_confocal_boundary](resources/3DUnet_confocal_boundary).
Pre-trained model weights available [here](https://oc.embl.de/index.php/s/61s67Mg5VQy7dh9/download?path=%2FArabidopsis-Ovules%2Funet_bce_dice_ds2x&files=best_checkpoint.pytorch).
In order to use the pre-trained model on your own data:
* download the `best_checkpoint.pytorch` from the above link
* add the path to the downloaded model and the path to your data in [test_config.yml](resources/3DUnet_confocal_boundary/test_config.yml)
* run `predict3dunet --config test_config.yml`
* optionally fine-tune the pre-trained model with your own data, by setting the `pre_trained` attribute in the YAML config to point to the `best_checkpoint.pytorch` path

The data used for training can be downloaded from the following OSF project:
* training set: https://osf.io/x9yns/
* validation set: https://osf.io/xp5uf/
* test set: https://osf.io/8jz7e/

Sample z-slice predictions on the test set (top: raw input , bottom: boundary predictions):

<img src="https://github.com/wolny/pytorch-3dunet/blob/master/resources/3DUnet_confocal_boundary/ovules_raw.png" width="400">
<img src="https://github.com/wolny/pytorch-3dunet/blob/master/resources/3DUnet_confocal_boundary/ovules_pred.png" width="400">

### Nuclei predictions for lightsheet images of Arabidopsis thaliana lateral root
Training/predictions configs can be found in [3DUnet_lightsheet_nuclei](resources/3DUnet_lightsheet_nuclei).
Pre-trained model weights available [here](https://oc.embl.de/index.php/s/61s67Mg5VQy7dh9/download?path=%2FLateral-Root-Primordia%2Funet_bce_dice_nuclei_ds1x&files=best_checkpoint.pytorch).
In order to use the pre-trained model on your own data:
* download the `best_checkpoint.pytorch` from the above link
* add the path to the downloaded model and the path to your data in [test_config.yml](resources/3DUnet_lightsheet_nuclei/test_config.yaml)
* run `predict3dunet --config test_config.yml`
* optionally fine-tune the pre-trained model with your own data, by setting the `pre_trained` attribute in the YAML config to point to the `best_checkpoint.pytorch` path

The training and validation sets can be downloaded from the following OSF project: https://osf.io/thxzn/

Sample z-slice predictions on the test set (top: raw input, bottom: nuclei predictions):

<img src="https://github.com/wolny/pytorch-3dunet/blob/master/resources/3DUnet_lightsheet_nuclei/root_nuclei_t30_raw.png" width="400">
<img src="https://github.com/wolny/pytorch-3dunet/blob/master/resources/3DUnet_lightsheet_nuclei/root_nuclei_t30_pred.png" width="400">


### 2D nuclei predictions for Kaggle DSB2018
The data can be downloaded from: https://www.kaggle.com/c/data-science-bowl-2018/data

Training/predictions configs can be found in [2DUnet_dsb2018](resources/2DUnet_dsb2018).

Sample predictions on the test image (top: raw input, bottom: nuclei predictions):

<img src="https://github.com/wolny/pytorch-3dunet/blob/master/resources/2DUnet_dsb2018/5f9d29d6388c700f35a3c29fa1b1ce0c1cba6667d05fdb70bd1e89004dcf71ed.png" width="400">
<img src="https://github.com/wolny/pytorch-3dunet/blob/master/resources/2DUnet_dsb2018/5f9d29d6388c700f35a3c29fa1b1ce0c1cba6667d05fdb70bd1e89004dcf71ed_predictions.png" width="400">

## Contribute
If you want to contribute back, please make a pull request.

## Cite
If you use this code for your research, please cite as:
```
@article {10.7554/eLife.57613,
article_type = {journal},
title = {Accurate and versatile 3D segmentation of plant tissues at cellular resolution},
author = {Wolny, Adrian and Cerrone, Lorenzo and Vijayan, Athul and Tofanelli, Rachele and Barro, Amaya Vilches and Louveaux, Marion and Wenzl, Christian and Strauss, Sören and Wilson-Sánchez, David and Lymbouridou, Rena and Steigleder, Susanne S and Pape, Constantin and Bailoni, Alberto and Duran-Nebreda, Salva and Bassel, George W and Lohmann, Jan U and Tsiantis, Miltos and Hamprecht, Fred A and Schneitz, Kay and Maizel, Alexis and Kreshuk, Anna},
editor = {Hardtke, Christian S and Bergmann, Dominique C and Bergmann, Dominique C and Graeff, Moritz},
volume = 9,
year = 2020,
month = {jul},
pub_date = {2020-07-29},
pages = {e57613},
citation = {eLife 2020;9:e57613},
doi = {10.7554/eLife.57613},
url = {https://doi.org/10.7554/eLife.57613},
keywords = {instance segmentation, cell segmentation, deep learning, image analysis},
journal = {eLife},
issn = {2050-084X},
publisher = {eLife Sciences Publications, Ltd},
}
```



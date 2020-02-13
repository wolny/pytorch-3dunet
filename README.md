[![DOI](https://zenodo.org/badge/149826542.svg)](https://doi.org/10.1101/2020.01.17.910562)
[![Build Status](https://travis-ci.com/wolny/pytorch-3dunet.svg?branch=master)](https://travis-ci.com/wolny/pytorch-3dunet)


# pytorch-3dunet

PyTorch implementation 3D U-Net and its variants:

- Standard 3D U-Net based on [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/abs/1606.06650) 
Özgün Çiçek et al.

- Residual 3D U-Net based on [Superhuman Accuracy on the SNEMI3D Connectomics Challenge](https://arxiv.org/pdf/1706.00120.pdf) Kisuk Lee et al.

The code allows for training the U-Net for both: **semantic segmentation** (binary and multi-class) and **regression** problems (e.g. de-noising, learning deconvolutions).

## 2D U-Net
Training the standard 2D U-Net is also possible, see [train_config_2d](resources/train_config_2d_boundary.yml) for example configuration. Just make sure to keep the singleton z-dimension in your H5 dataset (i.e. `(1, Y, X)` instead of `(Y, X)`) , cause data loading / data augmentation requires tensors of rank 3 always.

## Prerequisites
- Linux
- NVIDIA GPU
- CUDA CuDNN

### Running on Windows
The package has not been tested on Windows, however some reported using it on Windows. One thing to keep in mind:
when training with `CrossEntropyLoss`: the label type in the config file should be change from `long` to `int64`,
otherwise there will be an error: `RuntimeError: Expected object of scalar type Long but got scalar type Int for argument #2 'target'`.


## Supported Loss Functions

### Semantic Segmentation
- _BCEWithLogitsLoss_ (binary cross-entropy)
- _DiceLoss_ (standard `DiceLoss` defined as `1 - DiceCoefficient` used for binary semantic segmentation; when more than 2 classes are present in the ground truth, it computes the `DiceLoss` per channel and averages the values).
- _BCEDiceLoss_ (Linear combination of BCE and Dice losses, i.e. `alpha * BCE + beta * Dice`, `alpha, beta` can be specified in the `loss` section of the config)
- _CrossEntropyLoss_ (one can specify class weights via `weight: [w_1, ..., w_k]` in the `loss` section of the config)
- _PixelWiseCrossEntropyLoss_ (one can specify not only class weights but also per pixel weights in order to give more gradient to important (or under-represented) regions in the ground truth)
- _WeightedCrossEntropyLoss_ (see 'Weighted cross-entropy (WCE)' in the below paper for a detailed explanation; one can specify class weights via `weight: [w_1, ..., w_k]` in the `loss` section of the config)
- _GeneralizedDiceLoss_ (see 'Generalized Dice Loss (GDL)' in the below paper for a detailed explanation; one can specify class weights via `weight: [w_1, ..., w_k]` in the `loss` section of the config). 
Note: use this loss function only if the labels in the training dataset are very imbalanced e.g. one class having at least 3 orders of magnitude more voxels than the others. Otherwise use standard _DiceLoss_.


For a detailed explanation of some of the supported loss functions see:
[Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations](https://arxiv.org/pdf/1707.03237.pdf)
Carole H. Sudre, Wenqi Li, Tom Vercauteren, Sebastien Ourselin, M. Jorge Cardoso

### Regression
- _MSELoss_
- _L1Loss_
- _SmoothL1Loss_
- _WeightedSmoothL1Loss_ - extension of the _SmoothL1Loss_ which allows to weight the voxel values above (below) a given threshold differently


## Supported Evaluation Metrics

### Semantic Segmentation
- _MeanIoU_ - Mean intersection over union
- _DiceCoefficient_ - Dice Coefficient (computes per channel Dice Coefficient and returns the average)
If a 3D U-Net was trained to predict cell boundaries, one can use the following semantic instance segmentation metrics
(the metrics below are computed by running connected components on thresholded boundary map and comparing the resulted instances to the ground truth instance segmentation): 
- _BoundaryAveragePrecision_ - Average Precision applied to the boundary probability maps: thresholds the boundary maps given by the network, runs connected components to get the segmentation and computes AP between the resulting segmentation and the ground truth
- _AdaptedRandError_ - Adapted Rand Error (see http://brainiac2.mit.edu/SNEMI3D/evaluation for a detailed explanation)

If not specified `MeanIoU` will be used by default.


### Regression
- _PSNR_ - peak signal to noise ratio


## Getting Started

## Installation
- The easiest way to install `pytorch-3dunet` package is via conda:
```
conda create -n 3dunet -c conda-forge -c awolny python=3.7 pytorch-3dunet
conda activate 3dunet
```
After installation the following commands are accessible within the conda environment:
`train3dunet` for training the network and `predict3dunet` for prediction (see below).

- One can also install directly from source:
```
python setup.py install
```

### Installation tips
Make sure that the installed `pytorch` is compatible with your CUDA version, otherwise the training/prediction will fail to run on GPU. You can re-install `pytorch` compatible with your CUDA in the `3dunet` env by:
```
conda install -c pytorch torchvision cudatoolkit=<YOU_CUDA_VERSION> pytorch
```

## Train
Given that `pytorch-3dunet` package was installed via conda as described above, one can train the network by simply invoking:
```
train3dunet --config <CONFIG>
```
where `CONFIG` is the path to a YAML configuration file, which specifies all aspects of the training procedure.

See e.g. [train_config_ce.yaml](resources/train_config_ce.yaml) which describes how to train a standard 3D U-Net on a randomly generated 3D volume and random segmentation mask ([random_label3D.h5](resources/random3D.h5)) with cross-entropy loss (just a demo). 

In order to train on your own data just provide the paths to your HDF5 training and validation datasets in the [train_config_ce.yaml](resources/train_config_ce.yaml).
The HDF5 files should contain the raw/label data sets in the following axis order: `DHW` (in case of 3D) `CDHW` (in case of 4D).

One can monitor the training progress with Tensorboard `tensorboard --logdir <checkpoint_dir>/logs/` (you need `tensorflow` installed in your conda env), where `checkpoint_dir` is the path to the checkpoint directory specified in the config.

To try out training on randomly generated data right away, just checkout the repository and run: 
```
cd pytorch3dunet
train3dunet --config ../resources/train_config_ce.yaml # train with CrossEntropyLoss (segmentation)
#train3dunet --config ../resources/train_config_dice.yaml # train with DiceLoss (segmentation)
#train3dunet --config ../resources/train_config_regression.yaml # train with SmoothL1Loss (regression)
```

To try out a boundary prediction task given a sample 3D confocal volume of plant cells (cell membrane marker), run:
```
cd pytorch3dunet
train3dunet --config ../resources/train_boundary.yaml
```


### Training tips
When training with binary-based losses, i.e.: `BCEWithLogitsLoss`, `DiceLoss`, `BCEDiceLoss`, `GeneralizedDiceLoss`:
1. the label data has to be 4D (one target binary mask per channel).
If you have a 3D binary data (foreground/background), you can just change `ToTensor` transform for the label to contain `expand_dims: true`, see e.g. [train_config_dice.yaml](resources/train_config_dice.yaml).
2. `final_sigmoid=True` has to be present in the `model` section of the config, since every output channel gives the probability of the foreground.
When training with cross entropy based losses (`WeightedCrossEntropyLoss`, `CrossEntropyLoss`, `PixelWiseCrossEntropyLoss`) set `final_sigmoid=False` so that `Softmax` normalization is applied to the output.

## Prediction
Given that `pytorch-3dunet` package was installed via conda as described above, one can run the prediction via:
```
predict3dunet --config <CONFIG>
```

To run the prediction on randomly generated 3D volume (just for demonstration purposes) from [random_label3D.h5](resources/random3D.h5) and a network trained with cross-entropy loss:
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

## Sample configuration files

### Semantic segmentation
* [train with cross-entropy loss](resources/train_config_ce.yaml) / [predict using the network trained with cross-entropy loss](resources/test_config_ce.yaml)
* [train with Dice loss](resources/train_config_dice.yaml) / [predict using the network trained with Dice loss](resources/test_config_dice.yaml)
* [train using 4D input](resources/train_config_4d_input.yaml) / [predict on the 4D input](resources/test_config_4d_input.yaml)
* [train to predict cell boundaries from the confocal microscope](resources/train_config_boundary.yaml) / [predict using the network on the boundary classification task](resources/test_config_boundary.yaml)

### Regression
* [train on a random noise sample](resources/train_config_regression.yaml) / [predict using the network trained on a regression problem](resources/test_config_regression.yaml)

### 2D (semantic segmentation)
* [train to predict cell boundaries in 2D](resources/train_config_2d_boundary.yml) / [predict cell boundaries in 2D](resources/test_config_2d_boundary.yml)

## Contribute
If you want to contribute back, please make a pull request.

## Cite
If you use this code for your research, please cite as:
```
@article {Wolny2020.01.17.910562,
	author = {Wolny, Adrian and Cerrone, Lorenzo and Vijayan, Athul and Tofanelli, Rachele and Barro,
              Amaya Vilches and Louveaux, Marion and Wenzl, Christian and Steigleder, Susanne and Pape, 
              Constantin and Bailoni, Alberto and Duran-Nebreda, Salva and Bassel, George and Lohmann,
              Jan U. and Hamprecht, Fred A. and Schneitz, Kay and Maizel, Alexis and Kreshuk, Anna},
	title = {Accurate And Versatile 3D Segmentation Of Plant Tissues At Cellular Resolution},
	elocation-id = {2020.01.17.910562},
	year = {2020},
	doi = {10.1101/2020.01.17.910562},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2020/01/18/2020.01.17.910562}, 
	eprint = {https://www.biorxiv.org/content/early/2020/01/18/2020.01.17.910562.full.pdf},
	journal = {bioRxiv}
}
```



# pytorch-3dunet

PyTorch implementation of a standard 3D U-Net based on:

[3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/abs/1606.06650) 
Özgün Çiçek et al.

as well as Residual 3D U-Net based on:

[Superhuman Accuracy on the SNEMI3D Connectomics Challenge](https://arxiv.org/pdf/1706.00120.pdf) Kisuk Lee et al.

## Prerequisites
- Linux
- NVIDIA GPU
- CUDA CuDNN

## Getting Started

### Dependencies
- pytorch (0.4.1+)
- torchvision (0.2.1+)
- tensorboardx (1.6+)
- h5py
- scipy 
- scikit-image
- pytest

Setup a new conda environment with the required dependencies via:
```
conda create -n 3dunet pytorch torchvision tensorboardx h5py scipy scikit-image pyyaml pytest -c conda-forge -c pytorch
``` 
Activate newly created conda environment via:
```
source activate 3dunet
```

## Supported model architectures
- in order to train standard 3D U-Net specify `name: UNet3D` in the `model` section of the [config file](resources/train_config.yaml)
- in order to train Residual U-Net specify `name: ResidualUNet3D` in the `model` section of the [config file](resources/train_config.yaml)

## Supported Loss Functions
For a detailed explanation of the loss functions used see:
[Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations](https://arxiv.org/pdf/1707.03237.pdf)
Carole H. Sudre, Wenqi Li, Tom Vercauteren, Sebastien Ourselin, M. Jorge Cardoso

- _WeightedCrossEntropyLoss_ (see 'Weighted cross-entropy (WCE)' in the above paper for a detailed explanation)
- _CrossEntropyLoss_ (one can specify class weights via `--loss-weight <w_1 ... w_k>`)
- _PixelWiseCrossEntropyLoss_ (once can specify not only class weights but also per pixel weights in order to give more/less gradient in some regions of the ground truth)
- _BCEWithLogitsLoss_ (one can specify class weights via `--loss-weight <w_1 ... w_k>`)
- _DiceLoss_ standard Dice loss (see 'Dice Loss' in the above paper for a detailed explanation). Note: if your labels in the training dataset are not very imbalance
e.g. one class having at lease 3 orders of magnitude more voxels than the other use this instead of `GDL` since it worked better in my experiments.
- _GeneralizedDiceLoss_ (one can specify class weights via `--loss-weight <w_1 ... w_k>`)(see 'Generalized Dice Loss (GDL)' in the above paper for a detailed explanation)


## Supported Evaluation Metrics
- **MeanIoU** - Mean intersection over union
- **DiceCoefficient** - Dice Coefficient (computes per channel Dice Coefficient and returns the average)
- **BoundaryAveragePrecision** - Average Precision (normally used for evaluating instance segmentation, however it can be used when the 3D UNet is used to predict the boundary signal from the instance segmentation ground truth)
- **ARandScore** - Adjusted Rand Score

If not specified `iou` will be used by default.

## Train
E.g. fit to randomly generated 3D volume and random segmentation mask from [random_label3D.h5](resources/random_label3D.h5) run:
```
python train.py --config resources/train_config.yaml
```
See the [train_config.yaml](resources/train_config.yaml) for more info.

In order to train on your own data just provide the paths to your HDF5 training and validation datasets in the [train_config.yaml](resources/train_config.yaml).
The HDF5 files should contain the raw/label data sets in the following axis order: `DHW` (in case of 3D) `CDHW` (in case of 4D).

Monitor progress with Tensorboard `tensorboard --logdir ./3dunet/logs/ --port 8666` (you need `tensorboard` installed in your conda env).
![3dunet-training](https://user-images.githubusercontent.com/706781/45916217-9626d580-be62-11e8-95c3-508e2719c915.png)

### IMPORTANT
In order to train with `BinaryCrossEntropy`, `DiceLoss` or `GeneralizedDiceLoss` the label data has to be 4D (one target binary mask per channel). In case of `DiceLoss` and `GeneralizedDiceLoss` the final score is the average across channels.
`final_sigmoid=True` has to be present in the config when training the network with any of the above losses (and similarly `final_sigmoid=True` has to be passed to the `predict.py` if the network was trained with `final_sigmoid=True`)

## Test
Test on randomly generated 3D volume (just for demonstration purposes) from [random_label3D.h5](resources/random_label3D.h5). 
```
python predict.py --config resources/test_config.yaml
```
Prediction masks will be saved to `resources/random_label3D_probabilities.h5`.

In order to predict your own raw dataset provide the path to your model as well as paths to HDF5 test datasets in the [test_config.yaml](resources/test_config.yaml).

### IMPORTANT
In order to avoid block artifacts in the output prediction masks the patch predictions are averaged, so make sure that `patch/stride` params lead to overlapping blocks, e.g. `patch: [64 128 128] stride: [32 96 96]` will give you a 'halo' of 32 voxels in each direction.

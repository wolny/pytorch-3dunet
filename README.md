# pytorch-3dunet

PyTorch implementation of 3D U-Net based on:

[3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/abs/1606.06650)
Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf Ronneberger

## Prerequisites
- Linux
- NVIDIA GPU
- CUDA CuDNN

## Getting Started

### Dependencies
- pytorch (0.4.1+)
- torchvision (0.2.1+)
- tensorboardx (1.4+)
- h5py
- pytest

Setup a new conda environment with the required dependencies via:
```
conda create -n 3dunet pytorch torchvision tensorboardx h5py pytest -c conda-forge
``` 
Activate newly created conda environment via:
```
source activate 3dunet
```

## Train
```
usage: train.py [-h] --checkpoint-dir CHECKPOINT_DIR --in-channels IN_CHANNELS
                --out-channels OUT_CHANNELS [--interpolate]
                [--layer-order LAYER_ORDER] --loss LOSS
                [--loss-weight LOSS_WEIGHT [LOSS_WEIGHT ...]]
                [--epochs EPOCHS] [--iters ITERS] [--patience PATIENCE]
                [--learning-rate LEARNING_RATE] [--weight-decay WEIGHT_DECAY]
                [--validate-after-iters VALIDATE_AFTER_ITERS]
                [--log-after-iters LOG_AFTER_ITERS] [--resume RESUME]

UNet3D training

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint-dir CHECKPOINT_DIR
                        checkpoint directory
  --in-channels IN_CHANNELS
                        number of input channels
  --out-channels OUT_CHANNELS
                        number of output channels
  --interpolate         use F.interpolate instead of ConvTranspose3d
  --layer-order LAYER_ORDER
                        Conv layer ordering, e.g. 'crg' ->
                        Conv3D+ReLU+GroupNorm
  --loss LOSS           Which loss function to use. Possible values: [bce,
                        nll, dice]. Where bce - BinaryCrossEntropy (binary
                        classification only), nll - NegativeLogLikelihood
                        (multi-class classification), dice - DiceLoss (binary
                        classification only)
  --loss-weight LOSS_WEIGHT [LOSS_WEIGHT ...]
                        A manual rescaling weight given to each class in case
                        of NLLLoss. E.g. --loss-weight 0.3 0.3 0.4
  --epochs EPOCHS       max number of epochs (default: 500)
  --iters ITERS         max number of iterations (default: 1e5)
  --patience PATIENCE   number of epochs with no loss improvement after which
                        the training will be stopped (default: 20)
  --learning-rate LEARNING_RATE
                        initial learning rate (default: 0.0002)
  --weight-decay WEIGHT_DECAY
                        weight decay (default: 0.0001)
  --validate-after-iters VALIDATE_AFTER_ITERS
                        how many iterations between validations (default: 100)
  --log-after-iters LOG_AFTER_ITERS
                        how many iterations between tensorboard logging
                        (default: 100)
  --resume RESUME       path to latest checkpoint (default: none); if provided
                        the training will be resumed from that checkpoint
```


E.g. fit to randomly generated 3D volume and random segmentation mask from [random.h5](resources/random.h5) (see [train.py](train.py)):
```
python train.py --checkpoint-dir ~/3dunet --in-channels 1 --out-channels 2 --layer-order crg --loss nll --validate-after-iters 100 --log-after-iters 50 --epoch 50 --learning-rate 0.0002 --interpolate       
```
In order to resume training from the last checkpoint:
```
python train.py --resume ~/3dunet/last_checkpoint.pytorch --in-channels 1 --out-channels 2 --layer-order crg --loss nll --validate-after-iters 100 --log-after-iters 50 --epoch 50 --learning-rate 0.0002 --interpolate        
```
In order to train on your own data just provide paths to your HDF5 training and validation datasets (see [train.py](train.py)).
The HDF5 files should have the following scheme:
```
/raw - dataset containing the raw 3D/4D stack. The axis order has to be DxHxW/CxDxHxW
/label - dataset containing the label 3D stack with values 0..C (C - number of classes). The axis order has to be DxHxW.
```
Sometimes the problem to be solved requires to predict multiple channel binary masks. In that case the `label` dataset should be 4D and Binary Cross Entropy loss should be used during training:
```
/raw - dataset containing the raw 3D/4D stack. The axis order has to be DxHxW/CxDxHxW
/label - dataset containing the label 4D stack with values 0..1 (binary classification with C channels). The axis order has to be CxDxHxW.
```

Monitor progress with Tensorboard `tensorboard --logdir ~/3dunet/logs/ --port 8666` (you need `tensorboard` installed in your conda env).
![3dunet-training](https://user-images.githubusercontent.com/706781/45916217-9626d580-be62-11e8-95c3-508e2719c915.png)


## Test
```
usage: predict.py [-h] --model-path MODEL_PATH --in-channels IN_CHANNELS
                  --out-channels OUT_CHANNELS [--interpolate]
                  [--layer-order LAYER_ORDER] --loss LOSS

3D U-Net predictions

optional arguments:
  -h, --help            show this help message and exit
  --model-path MODEL_PATH
                        path to the model
  --in-channels IN_CHANNELS
                        number of input channels
  --out-channels OUT_CHANNELS
                        number of output channels
  --interpolate         use F.interpolate instead of ConvTranspose3d
  --layer-order LAYER_ORDER
                        Conv layer ordering, e.g. 'crg' ->
                        Conv3D+ReLU+GroupNorm
  --loss LOSS           Loss function used for training. Possible values:
                        [bce, nll, dice]. Has to be provided cause loss
                        determines the final activation of the model.
```

Test on randomly generated 3D volume (just for demonstration purposes) from [random.h5](resources/random.h5). 
See [predict.py](predict.py) for more info.
```
python predict.py --model-path ~/3dunet/best_checkpoint.pytorch --in-channels 1 --out-channels 2 --loss nll --interpolate --layer-order crg
```
Prediction masks will be saved to `~/3dunet/probabilities.h5`.

In order to predict your own raw dataset provide the path to your HDF5 test dataset (see [predict.py](predict.py)).
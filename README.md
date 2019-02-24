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
- tensorboardx (1.6+)
- h5py
- pytest

Setup a new conda environment with the required dependencies via:
```
conda create -n 3dunet pytorch torchvision tensorboardx h5py pyyaml pytest -c conda-forge -c pytorch
``` 
Activate newly created conda environment via:
```
source activate 3dunet
```

## Supported Losses
For a detailed explanation of the loss functions used see:
[Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations](https://arxiv.org/pdf/1707.03237.pdf)
Carole H. Sudre, Wenqi Li, Tom Vercauteren, Sebastien Ourselin, M. Jorge Cardoso

### Loss functions
- **wce** - _WeightedCrossEntropyLoss_ (see 'Weighted cross-entropy (WCE)' in the above paper for a detailed explanation)
- **ce** - _CrossEntropyLoss_ (one can specify class weights via `--loss-weight <w_1 ... w_k>`)
- **pce** - _PixelWiseCrossEntropyLoss_ (once can specify not only class weights but also per pixel weights in order to give more/less gradient in some regions of the ground truth)
- **bce** - _BCELoss_ (one can specify class weights via `--loss-weight <w_1 ... w_k>`)
- **dice** - _DiceLoss_ standard Dice loss (see 'Dice Loss' in the above paper for a detailed explanation). Note: if your labels in the training dataset are not very imbalance
e.g. one class having at lease 3 orders of magnitude more voxels than the other use this instead of `GDL` since it worked better in my experiments.
- **gdl** - _GeneralizedDiceLoss_ (one can specify class weights via `--loss-weight <w_1 ... w_k>`)(see 'Generalized Dice Loss (GDL)' in the above paper for a detailed explanation)


## Supported Evaluation Metrics
- **iou** - Mean intersection over union
- **dice** - Dice Coefficient (computes per channel Dice Coefficient and returns the average)
- **ap** - Average Precision (normally used for evaluating instance segmentation, however it can be used when the 3D UNet is used to predict the boundary signal from the instance segmentation ground truth)
- **rand** - Adjusted Rand Score

If not specified `iou` will be used by default.

## Train
```
usage: train.py [-h] [--config CONFIG] [--checkpoint-dir CHECKPOINT_DIR]
                [--in-channels IN_CHANNELS] [--out-channels OUT_CHANNELS]
                [--init-channel-number INIT_CHANNEL_NUMBER] [--interpolate]
                [--layer-order LAYER_ORDER] [--loss LOSS]
                [--loss-weight LOSS_WEIGHT [LOSS_WEIGHT ...]]
                [--ignore-index IGNORE_INDEX] [--eval-metric EVAL_METRIC]
                [--curriculum] [--final-sigmoid] [--epochs EPOCHS]
                [--iters ITERS] [--patience PATIENCE]
                [--learning-rate LEARNING_RATE] [--weight-decay WEIGHT_DECAY]
                [--validate-after-iters VALIDATE_AFTER_ITERS]
                [--log-after-iters LOG_AFTER_ITERS] [--resume RESUME]
                [--train-path TRAIN_PATH [TRAIN_PATH ...]]
                [--val-path VAL_PATH [VAL_PATH ...]]
                [--train-patch TRAIN_PATCH [TRAIN_PATCH ...]]
                [--train-stride TRAIN_STRIDE [TRAIN_STRIDE ...]]
                [--val-patch VAL_PATCH [VAL_PATCH ...]]
                [--val-stride VAL_STRIDE [VAL_STRIDE ...]]
                [--raw-internal-path RAW_INTERNAL_PATH]
                [--label-internal-path LABEL_INTERNAL_PATH]
                [--transformer TRANSFORMER]

UNet3D training

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the YAML config file
  --checkpoint-dir CHECKPOINT_DIR
                        checkpoint directory
  --in-channels IN_CHANNELS
                        number of input channels
  --out-channels OUT_CHANNELS
                        number of output channels
  --init-channel-number INIT_CHANNEL_NUMBER
                        Initial number of feature maps in the encoder path
                        which gets doubled on every stage (default: 64)
  --interpolate         use F.interpolate instead of ConvTranspose3d
  --layer-order LAYER_ORDER
                        Conv layer ordering, e.g. 'crg' ->
                        Conv3D+ReLU+GroupNorm
  --loss LOSS           Which loss function to use. Possible values: [bce, ce,
                        wce, dice]. Where bce - BinaryCrossEntropyLoss (binary
                        classification only), ce - CrossEntropyLoss (multi-
                        class classification), wce - WeightedCrossEntropyLoss
                        (multi-class classification), dice -
                        GeneralizedDiceLoss (multi-class classification)
  --loss-weight LOSS_WEIGHT [LOSS_WEIGHT ...]
                        A manual rescaling weight given to each class. Can be
                        used with CrossEntropy or BCELoss. E.g. --loss-weight
                        0.3 0.3 0.4
  --ignore-index IGNORE_INDEX
                        Specifies a target value that is ignored and does not
                        contribute to the input gradient
  --eval-metric EVAL_METRIC
                        Evaluation metric for semantic segmentation to be used
                        (default: iou)
  --curriculum          use simple Curriculum Learning scheme if ignore_index
                        is present
  --final-sigmoid       if True apply element-wise nn.Sigmoid after the last
                        layer otherwise apply nn.Softmax
  --epochs EPOCHS       max number of epochs (default: 500)
  --iters ITERS         max number of iterations (default: 1e5)
  --patience PATIENCE   number of validation rounds with no improvement after
                        which the training will be stopped (default: 20)
  --learning-rate LEARNING_RATE
                        initial learning rate (default: 0.0002)
  --weight-decay WEIGHT_DECAY
                        weight decay (default: 0)
  --validate-after-iters VALIDATE_AFTER_ITERS
                        how many iterations between validations (default: 100)
  --log-after-iters LOG_AFTER_ITERS
                        how many iterations between tensorboard logging
                        (default: 100)
  --resume RESUME       path to latest checkpoint (default: none); if provided
                        the training will be resumed from that checkpoint
  --train-path TRAIN_PATH [TRAIN_PATH ...]
                        paths to the training datasets, e.g. --train-path
                        <path1> <path2>
  --val-path VAL_PATH [VAL_PATH ...]
                        paths to the validation datasets, e.g. --val-path
                        <path1> <path2>
  --train-patch TRAIN_PATCH [TRAIN_PATCH ...]
                        Patch shape for used for training
  --train-stride TRAIN_STRIDE [TRAIN_STRIDE ...]
                        Patch stride for used for training
  --val-patch VAL_PATCH [VAL_PATCH ...]
                        Patch shape for used for validation
  --val-stride VAL_STRIDE [VAL_STRIDE ...]
                        Patch stride for used for validation
  --raw-internal-path RAW_INTERNAL_PATH
  --label-internal-path LABEL_INTERNAL_PATH
  --transformer TRANSFORMER
                        data augmentation class
```


E.g. fit to randomly generated 3D volume and random segmentation mask from [random_label3D.h5](resources/random_label3D.h5) (see [train.py](train.py)):
```
python train.py --checkpoint-dir ./3dunet --in-channels 1 --out-channels 2 --layer-order crg --loss ce --validate-after-iters 100 --log-after-iters 50 --epochs 50 --learning-rate 0.0002 --interpolate --train-path resources/random_label3D.h5 --val-path resources/random_label3D.h5 --train-patch 32 64 64 --train-stride 8 16 16 --val-patch 64 128 128 --val-stride 64 128 128     
```
In order to resume training from the last checkpoint:
```
python train.py --resume ./3dunet/last_checkpoint.pytorch --in-channels 1 --out-channels 2 --layer-order crg --loss ce --validate-after-iters 100 --log-after-iters 50 --epochs 50 --learning-rate 0.0002 --interpolate --train-path resources/random_label3D.h5 --val-path resources/random_label3D.h5 --train-patch 32 64 64 --train-stride 8 16 16 --val-patch 64 128 128 --val-stride 64 128 128       
```
One may also use a config file to do all of the above:
```
python train.py --config resources/train_config.yaml
```
In order to train on your own data just provide the paths to your HDF5 training and validation datasets in the [train_config.yaml](resources/train_config.yaml) or via the command line args.
The HDF5 files should contain the raw/labal datasets in the following axis order: `DHW` (in case of 3D) `CDHW` (in case of 4D)

Data augmentation is performed by default (see e.g. `StandardTransformer` in [transforms.py](augment/transforms.py) for more info).
If one wants to change/prevent data augmentation, one should provide their own implementation of `BaseTransformer`/use `BaseTransformer` (no augmentation).

Monitor progress with Tensorboard `tensorboard --logdir ./3dunet/logs/ --port 8666` (you need `tensorboard` installed in your conda env).
![3dunet-training](https://user-images.githubusercontent.com/706781/45916217-9626d580-be62-11e8-95c3-508e2719c915.png)

### IMPORTANT
In order to train with `BinaryCrossEntropy` the label data has to be 4D! (one target binary mask per channel). `--final-sigmoid` has to be given when training the network with `BinaryCrossEntropy`
(and similarly `--final-sigmoid` has to be passed to the `predict.py` if the network was trained with `--final-sigmoid`)

`DiceLoss` and `GeneralizedDiceLoss` support both 3D and 4D target (if the target is 3D it will be automatically expanded to 4D, i.e. each class in separate channel, before applying the loss).



## Test
```
usage: predict.py [-h] --model-path MODEL_PATH --in-channels IN_CHANNELS
                  --out-channels OUT_CHANNELS
                  [--init-channel-number INIT_CHANNEL_NUMBER] [--interpolate]
                  [--layer-order LAYER_ORDER] [--final-sigmoid] --test-path
                  TEST_PATH [--raw-internal-path RAW_INTERNAL_PATH] --patch
                  PATCH [PATCH ...] --stride STRIDE [STRIDE ...]

3D U-Net predictions

optional arguments:
  -h, --help            show this help message and exit
  --model-path MODEL_PATH
                        path to the model
  --in-channels IN_CHANNELS
                        number of input channels
  --out-channels OUT_CHANNELS
                        number of output channels
  --init-channel-number INIT_CHANNEL_NUMBER
                        Initial number of feature maps in the encoder path
                        which gets doubled on every stage (default: 64)
  --interpolate         use F.interpolate instead of ConvTranspose3d
  --layer-order LAYER_ORDER
                        Conv layer ordering, e.g. 'crg' ->
                        Conv3D+ReLU+GroupNorm
  --final-sigmoid       if True apply element-wise nn.Sigmoid after the last
                        layer otherwise apply nn.Softmax
  --test-path TEST_PATH
                        path to the test dataset
  --raw-internal-path RAW_INTERNAL_PATH
  --patch PATCH [PATCH ...]
                        Patch shape for used for prediction on the test set
  --stride STRIDE [STRIDE ...]
                        Patch stride for used for prediction on the test set
```

Test on randomly generated 3D volume (just for demonstration purposes) from [random_label3D.h5](resources/random_label3D.h5). 
See [predict.py](predict.py) for more info.
```
python predict.py --model-path ./3dunet/best_checkpoint.pytorch --in-channels 1 --out-channels 2 --interpolate --test-path resources/random_label3D.h5 --patch 64 128 128 --stride 32 64 64
```
Or use the config file:
```
python predict.py --config resources/test_config.yaml
```
Prediction masks will be saved to `./3dunet/random_label3D_probabilities.h5`.

In order to predict your own raw dataset provide the paths to your HDF5 test datasets in the [test_config.yaml](resources/test_config.yaml) or via the command line.

### IMPORTANT
In order to avoid block artifacts in the output prediction masks the patch predictions are averaged, so make sure that `patch/stride` params lead to overlapping blocks, e.g. `--patch 64 128 128 --stride 32 96 96` will give you a 'halo' of 32 voxels in each direction.

import importlib
import io
import logging
import os
import shutil
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from numpy import linalg as LA
from sklearn.decomposition import PCA

plt.ioff()
plt.switch_backend('agg')


def save_checkpoint(state, is_best, checkpoint_dir, logger=None):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.

    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    def log_info(message):
        if logger is not None:
            logger.info(message)

    if not os.path.exists(checkpoint_dir):
        log_info(
            f"Checkpoint directory does not exists. Creating {checkpoint_dir}")
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    log_info(f"Saving last checkpoint to '{last_file_path}'")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        log_info(f"Saving best checkpoint to '{best_file_path}'")
        shutil.copyfile(last_file_path, best_file_path)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied

    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer_state_dict'])

    return state


def save_network_output(output_path, output, logger=None):
    if logger is not None:
        logger.info(f'Saving network output to: {output_path}...')
    output = output.detach().cpu()[0]
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('predictions', data=output, compression='gzip')


loggers = {}


def get_logger(name, level=logging.INFO):
    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        loggers[name] = logger

        return logger


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


def find_maximum_patch_size(model, device):
    """Tries to find the biggest patch size that can be send to GPU for inference
    without throwing CUDA out of memory"""
    logger = get_logger('PatchFinder')
    in_channels = model.in_channels

    patch_shapes = [(64, 128, 128), (96, 128, 128),
                    (64, 160, 160), (96, 160, 160),
                    (64, 192, 192), (96, 192, 192)]

    for shape in patch_shapes:
        # generate random patch of a given size
        patch = np.random.randn(*shape).astype('float32')

        patch = torch \
            .from_numpy(patch) \
            .view((1, in_channels) + patch.shape) \
            .to(device)

        logger.info(f"Current patch size: {shape}")
        model(patch)


def remove_halo(patch, index, shape, patch_halo):
    """
    Remove `pad_width` voxels around the edges of a given patch.
    """
    assert len(patch_halo) == 3

    def _new_slices(slicing, max_size, pad):
        if slicing.start == 0:
            p_start = 0
            i_start = 0
        else:
            p_start = pad
            i_start = slicing.start + pad

        if slicing.stop == max_size:
            p_stop = None
            i_stop = max_size
        else:
            p_stop = -pad if pad != 0 else 1
            i_stop = slicing.stop - pad

        return slice(p_start, p_stop), slice(i_start, i_stop)

    D, H, W = shape

    i_c, i_z, i_y, i_x = index
    p_c = slice(0, patch.shape[0])

    p_z, i_z = _new_slices(i_z, D, patch_halo[0])
    p_y, i_y = _new_slices(i_y, H, patch_halo[1])
    p_x, i_x = _new_slices(i_x, W, patch_halo[2])

    patch_index = (p_c, p_z, p_y, p_x)
    index = (i_c, i_z, i_y, i_x)
    return patch[patch_index], index


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


class _TensorboardFormatter:
    """
    Tensorboard formatters converts a given batch of images (be it input/output to the network or the target segmentation
    image) to a series of images that can be displayed in tensorboard. This is the parent class for all tensorboard
    formatters which ensures that returned images are in the 'CHW' format.
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, name, batch):
        """
        Transform a batch to a series of tuples of the form (tag, img), where `tag` corresponds to the image tag
        and `img` is the image itself.

        Args:
             name (str): one of 'inputs'/'targets'/'predictions'
             batch (torch.tensor): 4D or 5D torch tensor
        """

        def _check_img(tag_img):
            tag, img = tag_img

            assert img.ndim == 2 or img.ndim == 3, 'Only 2D (HW) and 3D (CHW) images are accepted for display'

            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
            else:
                C = img.shape[0]
                assert C == 1 or C == 3, 'Only (1, H, W) or (3, H, W) images are supported'

            return tag, img

        tagged_images = self.process_batch(name, batch)

        return list(map(_check_img, tagged_images))

    def process_batch(self, name, batch):
        raise NotImplementedError


class DefaultTensorboardFormatter(_TensorboardFormatter):
    def __init__(self, skip_last_target=False, **kwargs):
        super().__init__(**kwargs)
        self.skip_last_target = skip_last_target

    def process_batch(self, name, batch):
        if name == 'targets' and self.skip_last_target:
            batch = batch[:, :-1, ...]

        tag_template = '{}/batch_{}/channel_{}/slice_{}'

        tagged_images = []

        if batch.ndim == 5:
            # NCDHW
            slice_idx = batch.shape[2] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                for channel_idx in range(batch.shape[1]):
                    tag = tag_template.format(name, batch_idx, channel_idx, slice_idx)
                    img = batch[batch_idx, channel_idx, slice_idx, ...]
                    tagged_images.append((tag, self._normalize_img(img)))
        else:
            # batch has no channel dim: NDHW
            slice_idx = batch.shape[1] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                tag = tag_template.format(name, batch_idx, 0, slice_idx)
                img = batch[batch_idx, slice_idx, ...]
                tagged_images.append((tag, self._normalize_img(img)))

        return tagged_images

    @staticmethod
    def _normalize_img(img):
        return np.nan_to_num((img - np.min(img)) / np.ptp(img))


def _pca_project(embeddings):
    assert embeddings.ndim == 3
    # reshape (C, H, W) -> (C, H * W) and transpose
    flattened_embeddings = embeddings.reshape(embeddings.shape[0], -1).transpose()
    # init PCA with 3 principal components: one for each RGB channel
    pca = PCA(n_components=3)
    # fit the model with embeddings and apply the dimensionality reduction
    flattened_embeddings = pca.fit_transform(flattened_embeddings)
    # reshape back to original
    shape = list(embeddings.shape)
    shape[0] = 3
    img = flattened_embeddings.transpose().reshape(shape)
    # normalize to [0, 255]
    img = 255 * (img - np.min(img)) / np.ptp(img)
    return img.astype('uint8')


class EmbeddingsTensorboardFormatter(DefaultTensorboardFormatter):
    def __init__(self, plot_variance=False, **kwargs):
        super().__init__(**kwargs)
        self.plot_variance = plot_variance

    def process_batch(self, name, batch):
        if name == 'predictions' or name == 'predictions1':
            return self._embeddings_to_rgb(batch)
        else:
            return super().process_batch(name, batch)

    def _embeddings_to_rgb(self, batch):
        assert batch.ndim == 5

        tag_template = 'embeddings/batch_{}/slice_{}'
        tagged_images = []

        slice_idx = batch.shape[2] // 2  # get the middle slice
        for batch_idx in range(batch.shape[0]):
            tag = tag_template.format(batch_idx, slice_idx)
            img = batch[batch_idx, :, slice_idx, ...]  # CHW
            # get the PCA projection
            rgb_img = _pca_project(img)
            tagged_images.append((tag, rgb_img))
            if self.plot_variance:
                cum_explained_variance_img = self._plot_cum_explained_variance(img)
                tagged_images.append((f'cumulative_explained_variance/batch_{batch_idx}', cum_explained_variance_img))

        return tagged_images

    def _plot_cum_explained_variance(self, embeddings):
        # reshape (C, H, W) -> (C, H * W) and transpose
        flattened_embeddings = embeddings.reshape(embeddings.shape[0], -1).transpose()
        # fit PCA to the data
        pca = PCA().fit(flattened_embeddings)

        plt.clf()
        # plot cumulative explained variance ratio
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance');
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        img = np.asarray(Image.open(buf)).transpose(2, 0, 1)
        return img


def get_tensorboard_formatter(config):
    if config is None:
        return DefaultTensorboardFormatter()

    class_name = config['name']
    m = importlib.import_module('pytorch3dunet.unet3d.utils')
    clazz = getattr(m, class_name)
    return clazz(**config)


class AbstractSamplePlotter:
    def __init__(self, plot_dir, **kwargs):
        self.plot_dir = plot_dir
        self.time = 0
        self.current_dir = None

    def update_current_dir(self):
        self.time += 1
        self.current_dir = os.path.join(self.plot_dir, str(self.time))
        os.makedirs(self.current_dir, exist_ok=False)

    def __call__(self, i, input, output, target, phase):
        assert self.current_dir is not None, 'update_current_dir() was not called'
        self.plot(i, input, output, target, phase)

    def plot(self, i, input, output, target, phase):
        raise NotImplementedError


class EmbeddingSamplePlotter(AbstractSamplePlotter):
    def __init__(self, plot_dir, epsilon, **kwargs):
        super().__init__(plot_dir, **kwargs)
        self.epsilon = epsilon

    def plot(self, i, input, embeddings, target, phase):
        if target.dim() == 5:
            # use 1st target channel
            target = target[:, 0, ...]

        if input.dim() == 5:
            input = input[:, 0, ...]

        input, embeddings, target = convert_to_numpy(input, embeddings, target)

        i_batch = 0
        # iterate over the batch
        for inp, emb, tar in zip(input, embeddings, target):
            seg = self._embedding_to_seg(emb, tar)
            file_path = os.path.join(self.current_dir, f'batch{i}_inst{i_batch}.png')
            plot_emb(inp, emb, seg, tar, file_path)
            i_batch += 1

    def _embedding_to_seg(self, embeddings, target):
        result = np.zeros(shape=embeddings.shape[1:], dtype=np.uint32)

        spatial_dims = (1, 2) if result.ndim == 2 else (1, 2, 3)

        labels, counts = np.unique(target, return_counts=True)
        for label, size in zip(labels, counts):
            # skip 0-label
            if label == 0:
                continue

            # get the mask for this instance
            instance_mask = (target == label)

            # mask out all embeddings not in this instance
            embeddings_per_instance = embeddings * instance_mask

            # compute the cluster mean
            mean_embedding = np.sum(embeddings_per_instance, axis=spatial_dims, keepdims=True) / size
            # compute the instance mask, i.e. get the epsilon-ball
            inst_mask = LA.norm(embeddings - mean_embedding, axis=0) < self.epsilon
            # save instance
            result[inst_mask] = label

        return result


def get_sample_plotter(config):
    class_name = config['name']
    m = importlib.import_module('pytorch3dunet.unet3d.utils')
    clazz = getattr(m, class_name)
    return clazz(**config)


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxSPATIAL label image to NxCxSPATIAL, where each label gets converted to its corresponding one-hot vector.
    It is assumed that the batch dimension is present.
    Args:
        input (torch.Tensor): 3D/4D input image
        C (int): number of channels/labels
        ignore_index (int): ignore index to be kept during the expansion
    Returns:
        4D/5D output torch.Tensor (NxCxSPATIAL)
    """
    assert input.dim() == 4

    # expand the input tensor to Nx1xSPATIAL before scattering
    input = input.unsqueeze(1)
    # create output tensor shape (NxCxSPATIAL)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)


def plot_segm(inp, seg, tar, file_path):
    """
    Saves predicted and ground truth segmentation into a PNG.
    """
    plt.clf()
    f, axarr = plt.subplots(1, 3)

    mid_z = seg.shape[0] // 2

    axarr[0].imshow(inp[mid_z])
    axarr[0].set_title('Input')

    axarr[1].imshow(seg[mid_z], cmap='prism')
    axarr[1].set_title('Predicted segmentation')

    axarr[2].imshow(tar[mid_z], cmap='prism')
    axarr[2].set_title('Ground truth')

    plt.savefig(file_path, bbox_inches='tight', transparent=True)


def plot_emb(inp, emb, seg, tar, file_path):
    plt.clf()
    f, axarr = plt.subplots(1, 4)

    mid_z = seg.shape[0] // 2

    rgb_emb = _pca_project(np.squeeze(emb))
    rgb_emb = np.transpose(rgb_emb, (1, 2, 0))

    axarr[0].imshow(inp[mid_z])
    axarr[0].set_title('Input')

    axarr[1].imshow(rgb_emb)
    axarr[1].set_title('Embeddings')

    axarr[2].imshow(seg[mid_z], cmap='prism')
    axarr[2].set_title('Predicted')

    axarr[3].imshow(tar[mid_z], cmap='prism')
    axarr[3].set_title('Ground truth')

    plt.savefig(file_path, bbox_inches='tight', transparent=True)


def convert_to_numpy(*inputs):
    """
    Coverts input tensors to numpy ndarrays

    Args:
        inputs (iteable of torch.Tensor): torch tensor

    Returns:
        tuple of ndarrays
    """

    def _to_numpy(i):
        assert isinstance(i, torch.Tensor), "Expected input to be torch.Tensor"
        return i.detach().cpu().numpy()

    return (_to_numpy(i) for i in inputs)

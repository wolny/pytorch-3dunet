import numpy as np
import os
import tifffile as tif
import h5py

def build_h5(weight_label=None):
    # set image and mask directories
    images_dir = '../dataset_exp50.1/train/raw/'
    masks_dir = '../dataset_exp50.1/train/labels/'

    # image and mask filenames sorted
    images_filenames = np.sort([f for f in os.listdir(images_dir) if f.endswith('.tif')])
    masks_filenames = np.sort([f for f in os.listdir(masks_dir) if f.endswith('.h5')])

    for i,j in zip(images_filenames, masks_filenames):
        image = tif.imread(files=images_dir + i)
        mask_h5 = h5py.File(masks_dir + j, 'r')
        mask = np.squeeze(mask_h5['exported_data'][...])
        print(i, image.shape, j, mask.shape)
        # assert image.shape = mask.shape
        h5_file = h5py.File('../dataset_exp50.1/train/hdf5/' + j, 'a')
        h5_file.create_dataset('raw', data=image)
        h5_file.create_dataset('label', data=mask)
        
        if weight_label:
            weight = np.copy(mask)
            weight[weight>1] = 1
            h5_file.create_dataset('weight', data=weight)
            print(weight.shape)
            print(np.unique(weight))

            
        

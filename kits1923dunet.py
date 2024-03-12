import sys
import os
import nibabel as nib
import h5py
import numpy as np

def convert_nifti_to_hdf5_for_case(case_dir, dest_dir, include_weight=False):
    """
    Convert NIfTI files for a case to HDF5 files in the specified destination directory,
    compatible with the StandardHDF5Dataset DataLoader.
    
    Parameters:
    - case_dir: Path to the directory containing 'imaging.nii.gz' and 'segmentation.nii.gz' for a case.
    - dest_dir: Path to the destination directory where 'data.hdf5' will be saved.
    - include_weight: Boolean, whether to include a placeholder weight dataset.
    """
    nifti_image_path = os.path.join(case_dir, 'imaging.nii.gz')
    nifti_label_path = os.path.join(case_dir, 'segmentation.nii.gz')
    
    if not os.path.exists(nifti_label_path):
        print(f"Label file missing for {case_dir}; skipping case.")
        return
    
    hdf5_path = os.path.join(dest_dir, 'data.hdf5')
    
    # Load the NIfTI files for image and label
    image = nib.load(nifti_image_path).get_fdata()
    label = nib.load(nifti_label_path).get_fdata()
    
    # Add a new axis to represent the single-channel dimension
    #image = image[np.newaxis, ...]  # Reshape from (Z, Y, X) to (C=1, Z, Y, X)
    #label = label[np.newaxis, ...]  # Reshape from (Z, Y, X) to (C=1, Z, Y, X)
    
    with h5py.File(hdf5_path, 'w') as hdf:
        hdf.create_dataset('raw', data=image, compression='gzip')
        hdf.create_dataset('label', data=label, compression='gzip')
        
        # Optionally include a placeholder weight dataset
        if include_weight:
            weight = np.ones_like(label, dtype=np.float32)  # Placeholder for actual weights
            hdf.create_dataset('weight', data=weight, compression='gzip')

def process_cases(origin_folder, dest_folder, include_weight=False):
    os.makedirs(dest_folder, exist_ok=True)
    case_dirs = [d for d in os.listdir(origin_folder) if os.path.isdir(os.path.join(origin_folder, d))]
    
    for case_dir in case_dirs:
        case_path = os.path.join(origin_folder, case_dir)
        case_dest_path = os.path.join(dest_folder, case_dir)
        
        if os.path.exists(os.path.join(case_path, 'segmentation.nii.gz')):
            os.makedirs(case_dest_path, exist_ok=True)
            convert_nifti_to_hdf5_for_case(case_path, case_dest_path, include_weight)
            print(f"Processed and saved data for {case_dir} to {case_dest_path}")
        else:
            print(f"Skipping {case_dir} due to missing label data.")

def main():
    if len(sys.argv) < 3:
        print("Usage: python main_script.py <source_directory> <destination_directory> [include_weight]")
        sys.exit(1)
    
    source_directory = sys.argv[1]
    destination_directory = sys.argv[2]
    include_weight = len(sys.argv) == 4 and sys.argv[3].lower() == 'true'
    
    process_cases(source_directory, destination_directory, include_weight)

if __name__ == "__main__":
    main()


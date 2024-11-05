import os
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import zoom
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import argparse


def load_pt_files(base_dir, dataset_names):
    datasets = {}
    for dataset_name in dataset_names:
        img_dir = '{}_MNI_to_TRs_minmax'.format(dataset_name)
        img_dir = os.path.join(base_dir, img_dir, 'img')

        if os.path.isdir(img_dir):
            object_folders = []
            for d in os.listdir(img_dir):
                object_folders.append(os.path.join(img_dir, d))
            datasets[dataset_name] = object_folders

    return datasets


def load_atlas_file(atlas_dir, filename):
    atlas_path = os.path.join(atlas_dir, filename)
    if os.path.exists(atlas_path):
        return nib.load(atlas_path).get_fdata()
    else:
        print(f"File {filename} not found in {atlas_dir}")
        return None


def resize_atlas_labels(atlas_labels, target_shape):
    zoom_factors = [t / a for t, a in zip(target_shape, atlas_labels.shape)]
    resized_labels = zoom(atlas_labels, zoom_factors, order=0)
    return resized_labels.astype(np.int32)


def load_frames_from_folder(folder_path):
    frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pt')], key=lambda x: int(x.split('_')[1].split('.')[0]))
    frames = []
    for f in frame_files:
        try:
            data = torch.load(os.path.join(folder_path, f)).numpy().squeeze()
        except:
            print('cannot read {}'.format(os.path.join(folder_path, f)))
            continue
        frames.append(data)
    return frames


def process_single_fmri_file(args):
    object_folder, atlases, output_dir, dataset_name = args
    
    base_name = os.path.basename(object_folder)
    all_exist = True
    exist_dict = {}
    for atlas_name in atlases.keys():
        output_file = os.path.join(output_dir, f"{base_name}_{atlas_name}.npy")
        if os.path.exists(output_file):
            print(f"Skipping {output_file} as it already exists.")
            exist_dict[atlas_name] = True
        else:
            exist_dict[atlas_name] = False
            all_exist = False

    if all_exist:
        return

    fmri_data = load_frames_from_folder(object_folder)
    num_frames = len(fmri_data)
    if num_frames < 10:
        print('{} has many broken data'.format(object_folder))
        return

    for atlas_name, atlas_data in atlases.items():
        if exist_dict[atlas_name]:
            continue
            
        if atlas_data.shape != fmri_data[0].shape:
            atlas_labels = resize_atlas_labels(atlas_data, fmri_data[0].shape)
            atlases[atlas_name] = atlas_labels
        else:
            atlas_labels = atlas_data
        unique_labels = np.unique(atlas_labels)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background (0)

        averages = []
        for frame in range(num_frames):
            frame_data = fmri_data[frame]
            frame_averages = []
            for label in unique_labels:
                mask = atlas_labels == label
                if np.any(mask):
                    mean_value = frame_data[mask].mean()
                    frame_averages.append(mean_value)
                else:
                    frame_averages.append(0)
            averages.append(frame_averages)

        averages = np.array(averages).T
        save_averages(averages, object_folder, atlas_name, output_dir, dataset_name)


def save_averages(averages, object_folder, atlas_name, output_dir, dataset_name):
    dataset_output_dir = output_dir
    os.makedirs(dataset_output_dir, exist_ok=True)
    base_name = os.path.basename(object_folder)
    output_file = os.path.join(dataset_output_dir, f"{base_name}_{atlas_name}.npy")
    np.save(output_file, averages)


def process_fmri_files(object_folders, atlases, output_dir, dataset_name, num_processes):
    args = [
        (object_folder, atlases, output_dir, dataset_name)
        for object_folder in object_folders
    ]

    if num_processes == 1:
        for arg in tqdm(args, desc=f'Processing {dataset_name}'):
            process_single_fmri_file(arg)
    else:
        num_files = len(object_folders)
        chunksize = max(1, num_files // (num_processes * 4))
        process_map(process_single_fmri_file, args, max_workers=num_processes, chunksize=chunksize)


def main():
    parser = argparse.ArgumentParser(description='Process fMRI data with selected atlases and datasets.')
    parser.add_argument('--atlas_names', type=str, nargs='+', required=True, help='Names of the atlases to use')
    parser.add_argument('--dataset_names', type=str, nargs='+', required=True, help='Names of the datasets to process')
    parser.add_argument('--fmri_dir', type=str, required=True, help='Directory containing fMRI data')
    parser.add_argument('--atlas_dir', type=str, required=True, help='Directory containing atlas data')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output files')
    parser.add_argument('--num_processes', type=int, default=1, help='Number of processes to use')
    args = parser.parse_args()

    # Load all atlas files
    atlases = {name: load_atlas_file(args.atlas_dir, name + '.nii.gz') for name in args.atlas_names}

    # Load fMRI datasets
    fmri_datasets = load_pt_files(args.fmri_dir, args.dataset_names)

    # Process each selected dataset with all atlases
    for dataset_name in args.dataset_names:
        object_folders = fmri_datasets[dataset_name]
        process_fmri_files(object_folders, atlases, args.output_dir, dataset_name, args.num_processes)


if __name__ == '__main__':
    main()

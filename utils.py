import os
import SimpleITK as sitk
import nibabel as nib
from pathlib import Path
import re
import pandas as pd
import yaml
import numpy as np
import matplotlib.pyplot as plt
import csv

# Function to load yaml configuration file
def load_config(config_name):
    """
    Load configuration file
    :param config_name:
    :return: configuration data
    """
    with open(config_name) as file:
        config = yaml.safe_load(file)

    return config


def load_mask(file):
    """
    Load masks from .gz files

    :param file: input filename
    :return: array of 2D masks (z, x, y)
    """
    mask = nib.load(file)
    mask = mask.get_fdata().transpose(2, 0, 1)
    mask = np.rot90(mask, axes=(1, 2))
    mask = mask.astype('int')

    return mask


def load_dicom(directory):
    """
    Load images from .dcm files

    :param directory: directory with .dcm files
    :return: array of 2D images (z, x, y)
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    image_itk = reader.Execute()
    image_zyx = sitk.GetArrayFromImage(image_itk).astype(np.int16)
    image_zyx = image_zyx.astype('int')

    return image_zyx


def preprocess_dataset(dir_original_images,
                       dir_original_masks,
                       dir_processed, save=True, output=False):
    """
    Load original images and masks from input directories
    and store them as numpy arrays

    :param dir_original_images: directory with input images
    :param dir_original_masks: directory with input masks
    :param dir_processed: directory for processed data
    :param save: save processed images and masks to .npz files and file with meta-data
    :param output: output processed images and masks

    :return array of images (z, x, y) and masks (z, x, y) if param output=True
    """

    images = np.empty(shape=(0, 512, 512)).astype('int')
    masks = np.empty(shape=(0, 512, 512)).astype('int')

    images_folders = [str(name) for name in
                      list(Path(os.path.join(dir_original_images)).rglob("**"))
                      if os.listdir(name)[0].endswith('.dcm')]

    masks_values = []

    for folder in images_folders:

        images = np.concatenate((images, load_dicom(folder)), axis=0)

        patient = re.findall(r'LUNG[0-9]*-[0-9]*', folder)[0]

        mask_file = str(list(Path(os.path.join(dir_original_masks, patient))
                             .rglob("*.gz"))[0])

        masks_batch = load_mask(mask_file)

        if save:
            masks_values = masks_values + [[patient, 1] if mask.sum() > 0 else [patient, 0]
                                           for mask in masks_batch]

        masks = np.concatenate((masks, masks_batch), axis=0)

    print(f'Images read: {images.shape[0]}')
    print(f'Masks read: {masks.shape[0]}')

    if save:
        np.savez_compressed(os.path.join(dir_processed, 'images'), *[image for image in images])
        np.savez_compressed(os.path.join(dir_processed, 'masks'), *[mask for mask in masks])

        masks_df = pd.DataFrame(data=masks_values, columns=['Patient', 'Effusion'])

        masks_df.to_csv(os.path.join(dir_processed, 'meta_file.csv'),
                        sep='\t', index_label='index')

    if output:
        return images, masks


def show_dice_example(mask1, mask2, dice_coef):
    """
    Show plots with two binary masks and dice coefficient between them
    :param mask1:  binary mask [n, z, x, y]
    :param mask2: binary mask [n, z,  x, y]
    :param dice_coef: dice coefficient
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    plt.suptitle(f'DICE coefficient: {dice_coef:.2f}')

    ax[0].imshow(mask1[0].permute(1, 2, 0))
    ax[1].imshow(mask2[0].permute(1, 2, 0))

    ax[0].set_title('Mask 1')
    ax[1].set_title('Mask 2')

    plt.show()


def save_history(history_dict, output_folder):
    """
    Save training history file to csv
    :param history_dict:  dictionary with training history
    :param output_folder:  folder to write csv
    """
    try:
        with open(os.path.join(output_folder, 'train_history.csv'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=history_dict.keys())
            writer.writeheader()
            for data in history_dict:
                writer.writerow(data)
    except IOError:
        print("I/O error")
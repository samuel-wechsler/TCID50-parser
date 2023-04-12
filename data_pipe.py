"""
data_pipe.py

The data_pipe module contains functions for loading image data and
preparing them for use in a machine learning model. The functions
in this module can be used to load images from a directory or a file,
resize them, and return them in the form of tensors. The module also
contains a command-line interface that allows users to specify the 
input and output directories for the image files.
"""
import os
import csv
from pathlib import Path
import sys
import shutil
import random
import numpy as np
import gmpy2
from datetime import datetime
from tqdm import tqdm

from wbns import wbns
from skimage import exposure
from skimage.io import imread, imsave
import tifffile as tif
from PIL import Image
import cv2
from itertools import product

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa
import tensorflow as tf
import tensorflow_io as tfio

import pandas as pd

IMG_HEIGHT, IMG_WIDTH = 128, 128


def load_and_prep_image(filename, img_shape=IMG_WIDTH):
    """
    This function reads an image from a file, turns it into a
    tensor and resizes it to a specific shape.
    """
    # Read in target file (an image)
    img = tf.io.read_file(filename)

    # Decode the read file into a tensor & ensure 3 colour channels
    if filename.endswith("tif"):
        img = tfio.experimental.image.decode_tiff(img)
        img = tfio.experimental.color.rgba_to_rgb(img)
    else:
        img = tf.image.decode_image(img, channels=3)

    # Resize the image
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = tf.expand_dims(img, axis=0)

    return img


def get_file_paths(root_dir):
    """
    This function returns all file paths that are in the root_dir of one of its
    subdirectories.
    """
    skips = ['.DS_Store', '.DS_S_i.png', '._', '.docx']
    file_paths = []

    # walk through a directory
    for dirpath, dirnames, filenames in os.walk(root_dir):

        for filename in filenames:
            path = os.path.join(dirpath, filename)
            if filename not in skips and not any([skip in path for skip in skips]):
                file_paths.append(path)

    return file_paths


def load_data_from_dir(data_dir):
    """
    This function loads the images from a data directory and their corresponding classficications
    from a given data_dir and resizes them and then resizes the images. The labels and the
    corresponding image data are returned in two lists.
    """

    # certain files and directory that aren't loaded
    skip = ['.DS_Store', '.DS_S_i.png', 'Session', '._',
            'datasets/matura_data/merge', 'datasets/matura_data/PhaseContrast']  # 'datasets/Laloli_et_all2022_raw_images',

    images = []
    labels = []

    file_paths = get_file_paths(data_dir)

    for path in tqdm(file_paths, total=len(file_paths)):
        filename, ext = os.path.splitext(path)

        # ignore files that don't exist, are in the skip list, who's directory is in the skip list or don't have the filetype png
        if (not os.path.isfile(path) or filename in skip or any([ski in path for ski in skip])) is False:

            # parse label
            label = 0 if 'not_infected' in path else 1
            labels.append(label)

            # parse image as ndarray
            im = cv2.imread(path)

            # resize image
            resizeIM = cv2.resize(im, (IMG_HEIGHT, IMG_WIDTH))
            images.append(resizeIM)

    return (images, labels)


def load_data_from_classfile(class_file, denoise_model_path=None, grayscale=False):
    """
    This function loads the images from a data directory and their corresponding classficications
    from a given data_dir and resizes them and then resizes the images. The labels and the
    corresponding image data are returned in two lists.
    """
    print("loading data...")

    # load model
    if denoise_model_path is not None:
        denoise_model = tf.keras.models.load_model(denoise_model_path)

    # certain files and directory that aren't loaded
    skip = ['.DS_Store', '.DS_S_i.png', 'Session',
            'datasets/matura_data/merge', 'datasets/matura_data/PhaseContrast']  # 'datasets/Laloli_et_all2022_raw_images',

    images = []
    labels = []

    seen = []

    class_file = open(class_file, 'r')
    reader = csv.reader(class_file, delimiter=';')
    rows = [(path, label) for (path, label) in reader]

    for path, label in tqdm(rows, total=len(rows)):
        label = int(label.replace('\n', ""))

        # ignore files that don't exist, are in the skip list, who's directory is in the skip list
        # or have ._ in their name
        if os.path.isfile(path) and path not in seen and '._' not in path:

            # parse label
            labels.append(label)

            # parse image as ndarray
            im = cv2.imread(path)

            # resize image
            resizeIM = cv2.resize(im, (IMG_HEIGHT, IMG_WIDTH))

            images.append(resizeIM)

            seen.append(path)

    return (images, labels)


def reduce_dims(input):
    known_axes = [i for i, size in enumerate(input.shape) if size == 1]
    y = tf.squeeze(input, axis=known_axes)
    return tf.squeeze(input, axis=known_axes)


def load_data_df_from_dir(data_dir, limit=np.inf):
    """
    This function loads all images (except those listed in skip list) and returns
    their paths and labels in two lists.
    """
    # certain files and directory that aren't loaded
    skip = ['.DS_Store', '.DS_S_i.png', '._']

    files = []
    labels = []

    file_paths = get_file_paths(data_dir)
    c = 0

    for path in tqdm(file_paths, total=len(file_paths)):
        filename, ext = os.path.splitext(os.path.split(path))

        # ignore files that don't exist, are in the skip list, or who's directory is in the skip list
        if (os.path.isfile(path)) and (filename not in skip) and (not any([frag in path for frag in skip])) and c < limit:
            # parse label
            label = "not infected" if 'not_infected' in path else "infected"
            labels.append(label)

            files.append(path)

            c += 1

    # df = pd.DataFrame(list(zip(files, labels)), columns=["filenames", "labels"])
    return files, labels


def load_data_df_from_class(classfile, limit=np.inf):
    """
    This function loads all images (except those listed in skip list) and returns
    their paths and labels in two lists.
    """
    # certain files and directory that aren't loaded
    skip = ['.DS_Store', '.DS_S_i.png', '._']

    files = []
    labels = []

    classfile = open(classfile, 'r')
    c = 0

    for line in classfile:
        entries = line.split(";")
        filepath = entries[0]
        filename = os.path.basename(filepath)
        classification = int(entries[1].replace("\n", ""))

        # ignore files that don't exist, are in the skip list, or who's directory is in the skip list
        if (os.path.isfile(filepath)) and (filename not in skip) and (not any([frag in filepath for frag in skip])) and c < limit:
            # parse label and filepath
            label = "infected" if classification else "not infected"
            labels.append(label)
            files.append(filepath)
            c += 1

    # df = pd.DataFrame(list(zip(files, labels)), columns=["filenames", "labels"])
    return files, labels


def load_fmd_data(data_dir, img_shape=900):
    """
    This function returns two lists of noisy and denoised images (as ndarrays) that were
    found in a given data directory.
    """

    # skip ground truth and raw data
    skips = ["gt", "raw", '._']

    # lists for noisy and denoised
    noisy = []
    denoised = []

    t_c = 0

    # walk through all files in a directory
    for dirpath, dirnames, filenames in os.walk(data_dir):

        # choose 4 random images
        filenames = random.choices(filenames, k=1)

        for filename in filenames:
            # get path of noisy image
            noisy_path = os.path.join(dirpath, filename)

            # only parse noisy files, skip others
            if any([skip in noisy_path for skip in skips]) or not filename.endswith(".png"):
                print(f"skipped {filename}")
                continue

            # if t_c > 99:
            #     return (noisy, denoised)

            t_c += 1
            print(f"{t_c} parsing {filename}")

            # find ground truth (denoised) analogue
            parent0 = Path(os.path.abspath(noisy_path)).parents[0]
            parent2 = Path(os.path.abspath(noisy_path)).parents[2]
            img_nb = os.path.basename(parent0)
            img_dir = os.path.join(parent2, "gt", img_nb)
            filename = [
                png for png in os.listdir(img_dir) if png.endswith('.png')
            ]
            denoised_path = os.path.join(img_dir, filename[0])

            # parse images as ndarray
            noisy_im = cv2.imread(noisy_path)
            denoised_im = cv2.imread(denoised_path)

            # resize images
            resize_noisy_im = cv2.resize(noisy_im, (img_shape, img_shape))
            resize_denoised_im = cv2.resize(
                denoised_im, (img_shape, img_shape))

            noisy.append(resize_noisy_im)
            denoised.append(resize_denoised_im)

    return (noisy, denoised)


def parse_files_gamma(input_dir, output_dir, filetype="png"):
    """
    This function parses all GFP images from the a given input_dir, perfoms
    an gamma correction to adjust for auto fluorescence of cells and then
    moves them to an output_dir.
    """

    # walk through a directory
    for dirpath, dirnames, filenames in os.walk(input_dir):

        for filename in filenames:

            # check if file is GFP image
            if filename.endswith('.tif') and "GFP" in filename:

                # parse filepath of image
                src = os.path.join(dirpath, filename)

                # parse new filename of image
                paths = dirpath.split('/')
                infos = [path.split('_') for path in paths]
                meta = []
                meta.extend(infos[2][-3:])
                meta.extend(infos[3][-2:])
                filename = "_".join(meta) + "_GFP." + filetype

                dst = os.path.join(output_dir, filename)

                print(f"converting {filename}")

                # save image with lower brightness
                image = imread(src)
                image_dark = exposure.adjust_gamma(image, gamma=2, gain=1)
                imsave(dst, image_dark)


def parse_files_wbns_from_dir(input_dir, output_dir, filetype="tif", limit=np.inf):
    """
    This function parses all GFP images from the a given input_dir, perfoms
    an Wavelet-based Background Subtraction (WBNS) to adjust for auto fluorescence
    noise of cells and background and then moves them to an output_dir.
    """
    skips = ['.DS_Store', '.DS_S_i.png', '._']

    # walk through a directory
    for dirpath, dirnames, filenames in os.walk(input_dir):

        for filename in filenames:
            path = os.path.join(dirpath, filename)

            # check if file is GFP image
            if filename not in skips and not any([skip in path for skip in skips]):

                print(f"Converting {filename}")

                # parse filepath of image
                src = os.path.join(dirpath, filename)

                # convert filetype
                filename = os.path.splitext(os.path.basename(filename))[
                    0] + "." + filetype

                if "M" in filename:
                    dst = os.path.join(output_dir, "not_infected", filename)
                else:
                    dst = os.path.join(output_dir, "infected", filename)

                # print(src)
                # print(dst)

                image = wbns(src)
                tif.imsave(dst, image, bigtiff=False)


def parse_files_wbns_from_class(classfile, output_dir, filetype="tif", limit=np.inf):
    """
    This function parses all GFP images from the a given input_dir, perfoms
    an Wavelet-based Background Subtraction (WBNS) to adjust for auto fluorescence
    noise of cells and background and then moves them to an output_dir.
    """
    skips = ['.DS_Store', '.DS_S_i.png', '._']
    added = []

    # read classification file
    file = open(classfile, 'r', newline='\n')
    reader = csv.reader(file, delimiter=';')

    counter = 0

    for row in reader:
        src = row[0]
        filename = os.path.basename(src)
        label = int(row[1])

        # check if file is GFP image
        if filename not in skips and not any([skip in src for skip in skips]) and counter < limit:
            t1 = datetime.now()

            filename_no_ext = os.path.splitext(os.path.basename(filename))[0]
            i = added.count(filename_no_ext)
            added.append(filename_no_ext)

            filename = filename_no_ext + f"_{i}." + filetype

            if label:
                dst = os.path.join(output_dir, "infected", filename)
            else:
                dst = os.path.join(output_dir, "not_infected", filename)

            image = wbns(src)
            tif.imsave(dst, image, bigtiff=False)

            t2 = datetime.now()

            print(f"Converted {filename} in {t2 - t1}")

            counter += 1


# parse_files_wbns_from_class(
#     "classifications/classification.txt", "/Volumes/T7/tcid50_datasets/Lukas_Probst_IDV/Lukas_Probst_wbns_tif")


def parse_file_from_class(class_file, ouput_dir):
    """
    This function copies all images listed in a classifications.txt file into the
    output directory and subfolders "infected" or "not_infected" depending on
    their labels.
    """
    # read classifications.txt
    class_file = open(class_file, 'r')

    for line in class_file:
        # parse source file path and label
        data = line.replace("\n", "").split(';')
        src = data[0]
        label = "infected" if data[1] == "1" else "not_infected"

        # parse destination directory
        filename = os.path.basename(src)
        dst = os.path.join(ouput_dir, label, filename)

        # copy image file into dst directory
        print(f"parsing {filename}")
        if filename in os.listdir(os.path.join(ouput_dir, label)):
            print("double")
        shutil.copy(src, dst)


def tile(filename, file_count, input_dir, output_dir, n):
    """
    splits image into n tiles
    """
    if not gmpy2.is_square(n):
        raise Exception("n must be a square number")

    # read image
    img = Image.open(os.path.join(input_dir, filename))

    # get size of tiles
    d = int(img.size[0] / np.sqrt(n))

    name, ext = os.path.splitext(filename)
    w, h = img.size

    grid = product(range(0, h-h % d, d), range(0, w-w % d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        dst = os.path.join(output_dir, f"{name}_{i}_{j}_{file_count}{ext}")
        img.crop(box).save(dst)


def tile_dir(data_dir, output_dir, n):
    skips = ['.DS_Store', '.DS_S_i.png', '._', '.docx']
    added = []

    # walk through a directory
    for dirpath, dirnames, filenames in os.walk(data_dir):

        for filename in filenames:
            path = os.path.join(dirpath, filename)

            if filename not in skips and not any([skip in path for skip in skips]):
                print(f"tiling {filename}")

                added.append(filename)

                file_count = added.count(filename)
                tile(filename, file_count, dirpath, output_dir, n)


def square_images(src, dst, s):
    """
    """
    filepaths = get_file_paths(src)

    for filepath in tqdm(filepaths, total=len(filepaths)):
        path, filename = os.path.split(filepath)
        filename, ext = os.path.splitext(filename)

        img = Image.open(filepath)

        w, h = img.size

        # crop upper left
        box_ul = (0, 0, s, s)
        img.crop(box_ul).save(os.path.join(dst, f"{filename}_ul{ext}"))

        # crop upper right
        box_ur = (w-s, 0, w, s)
        img.crop(box_ur).save(os.path.join(dst, f"{filename}_ur{ext}"))

        # crop lower left
        box_ll = (0, h-s, s, h)
        img.crop(box_ll).save(os.path.join(dst, f"{filename}_ll{ext}"))

        # crop lower right
        box_lr = (w-s, h-s, w, h)
        img.crop(box_lr).save(os.path.join(dst, f"{filename}_lr{ext}"))


def replace_filetype(data_dir, old, new):
    """
    This function changes all images filetypes in data_dir from old to new and deletes
    the old image (!).
    """

    # walk through a directory
    for dirpath, dirnames, filenames in os.walk(data_dir):

        for filename in filenames:
            if filename.endswith(old):
                # load image
                src = os.path.join(dirpath, filename)
                image = Image.open(src)

                # parse new filename and dst path
                new_filename = filename.replace(old, "") + new
                dst = os.path.join(dirpath, new_filename)

                print(src)
                print(dst)

                # save image
                image.save(dst)

                # delete old one
                os.remove(src)

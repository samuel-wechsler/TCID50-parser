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
from pathlib import Path
import sys
import shutil
import random

from wbns import wbns
from skimage import exposure
from skimage.io import imread, imsave
import tifffile as tif
from PIL import Image
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa
import tensorflow as tf
import tensorflow_io as tfio

import pandas as pd

IMG_HEIGHT, IMG_WIDTH = 900, 900


def load_and_prep_image(filename, img_shape=128):
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


def load_data_from_dir(data_dir):
    """
    This function loads the images from a data directory and their corresponding classficications
    from a given data_dir and resizes them and then resizes the images. The labels and the
    corresponding image data are returned in two lists.
    """

    # certain files and directory that aren't loaded
    skip = ['.DS_Store', '.DS_S_i.png', 'Session',
            'datasets/matura_data/merge', 'datasets/matura_data/PhaseContrast']  # 'datasets/Laloli_et_all2022_raw_images',

    images = []
    labels = []

    # walk through all files in a directory
    for dirpath, dirnames, filenames in os.walk(data_dir):

        for filename in filenames:
            # get path of file
            path = os.path.join(dirpath, filename)

            # ignore files that don't exist, are in the skip list, who's directory is in the skip list or don't have the filetype png
            if (not os.path.isfile(path) or filename in skip or any([ski in dirpath for ski in skip]) or not filename.endswith(".png")) is False:

                # parse label
                label = 0 if 'not_infected' in dirpath else 1
                labels.append(label)

                # parse image as ndarray
                im = cv2.imread(path)

                # resize image
                resizeIM = cv2.resize(im, (IMG_HEIGHT, IMG_WIDTH))
                images.append(resizeIM)

                print(f"parsing file {filename} --> {label}")

    return (images, labels)


def load_data_from_classfile(class_file, denoise_model_path=None, grayscale=False):
    """
    This function loads the images from a data directory and their corresponding classficications
    from a given data_dir and resizes them and then resizes the images. The labels and the
    corresponding image data are returned in two lists.
    """
    # load model
    if denoise_model_path is not None:
        denoise_model = tf.keras.models.load_model(denoise_model_path)

    # certain files and directory that aren't loaded
    skip = ['.DS_Store', '.DS_S_i.png', 'Session',
            'datasets/matura_data/merge', 'datasets/matura_data/PhaseContrast']  # 'datasets/Laloli_et_all2022_raw_images',

    images = []
    labels = []

    seen = []

    with open(class_file, 'r') as f:
        c = 0

        for line in f:
            entries = line.split(';')
            path = entries[0]
            label = entries[1]

            # ignore files that don't exist, are in the skip list, who's directory is in the skip list
            # or have ._ in their name
            if os.path.isfile(path) and path not in seen and '._' not in path:

                # parse label
                labels.append(label)

                # load and denoise image
                im = load_and_prep_image(path, img_shape=300)

                if grayscale:
                    resizeIM = cv2.cvtColor(resizeIM, cv2.COLOR_BGR2GRAY)

                images.append(im)

                seen.append(path)

                c += 1

                if c % 20 == 0:
                    print(f"{c} files parsed and denoised")

    return (images, labels)


def reduce_dims(input):
    known_axes = [i for i, size in enumerate(input.shape) if size == 1]
    y = tf.squeeze(input, axis=known_axes)
    return tf.squeeze(input, axis=known_axes)


def load_data_df(data_dir):
    """
    This function loads all images (except those listed in skip list) and returns
    their paths and labels in two lists.
    """
    # certain files and directory that aren't loaded
    skip = ['.DS_Store', '.DS_S_i.png', '._']

    files = []
    labels = []

    # walk through all files in a directory
    for dirpath, dirnames, filenames in os.walk(data_dir):

        for filename in filenames:
            # get path of file
            path = os.path.join(dirpath, filename)

            # ignore files that don't exist, are in the skip list, or who's directory is in the skip list
            if (os.path.isfile(path)) and (filename not in skip) and (not any([frag in path for frag in skip])):
                # parse label
                print('._' in path)
                label = "not infected" if 'not_infected' in dirpath else "infected"
                labels.append(label)

                files.append(os.path.join(dirpath, filename))

    # df = pd.DataFrame(list(zip(files, labels)), columns=["filenames", "labels"])
    return files, labels


def load_fmd_data(data_dir):
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
        filenames = random.choices(filenames, k=4)

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
            resize_noisy_im = cv2.resize(noisy_im, (IMG_HEIGHT, IMG_WIDTH))
            resize_denoised_im = cv2.resize(denoised_im,
                                            (IMG_HEIGHT, IMG_WIDTH))

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


def parse_files_wbns(input_dir, output_dir, filetype="tif"):
    """
    This function parses all GFP images from the a given input_dir, perfoms
    an Wavelet-based Background Subtraction (WBNS) to adjust for auto fluorescence
    noise of cells and background and then moves them to an output_dir.
    """

    # walk through a directory
    for dirpath, dirnames, filenames in os.walk(input_dir):

        for filename in filenames:

            # check if file is GFP image
            if filename.endswith('.tif') and "GFP" in filename:

                print(f"Converting {filename}")

                # parse filepath of image
                src = os.path.join(dirpath, filename)

                # convert filetype
                filename = filename[:-3] + filetype

                if "M" in filename:
                    dst = os.path.join(output_dir, "not_infected", filename)
                else:
                    dst = os.path.join(output_dir, "infected", filename)

                # print(src)
                # print(dst)

                image = wbns(src)
                tif.imsave(dst, image, bigtiff=False)


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
        shutil.copy(src, dst)


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

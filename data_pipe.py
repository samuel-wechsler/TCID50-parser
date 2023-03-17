"""
data_pipe.py

This python module contains functions that are used to load image data
"""

import os
import sys
import shutil

import pandas as pd

import tensorflow as tf
import cv2
from PIL import Image
import tifffile as tif
from skimage.io import imread, imsave
from skimage import exposure

from wbns import wbns

IMG_HEIGHT, IMG_WIDTH = 128, 128


def main():
    commands = {"-help": None,
                "-parse": ("path/to/input/dir", "path/to/output/dir", "[filetype]")}

    # check validity of command line argument
    if len(sys.argv) >= 5:
        sys.exit("not a valid command: python data_pipe.py -help")

    if sys.argv[1] == '-help':
        # loop trhough and display all commands
        print("See a list of all commands:")
        for com in commands.keys():
            if commands[com] is not None:
                print(com, " ".join(list(commands[com])))
            else:
                print(com)

    elif sys.argv[1] == '-parse':
        if len(sys.argv) not in [4, 5]:
            sys.exit("not a valid command: python data_pipe.py -help")

        input_dir = sys.argv[2]
        output_dir = sys.argv[3]

        # in case of optional filetype argument
        if len(sys.argv) == 5:
            parse_files_gamma(input_dir, output_dir, filetype=sys.argv[4])
        else:
            parse_files_gamma(input_dir, output_dir)

    # again, check validity of command
    else:
        sys.exit("not a valid command: python data_pipe.py -help")


def load_and_prep_image(filename, img_shape=128):
    """
    Reads an image from filename, turns it into a tensor
    and reshapes it to (img_shape, img_shape, colour_channel).
    """
    # Read in target file (an image)
    img = tf.io.read_file(filename)

    # Decode the read file into a tensor & ensure 3 colour channels
    img = tf.image.decode_image(img, channels=3)

    # Resize the image
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = tf.expand_dims(img, axis=0)

    return img


def load_data(data_dir):
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


def load_data_df(data_dir):
    """
    This function loads all images (except those listed in skip list) and returns
    their paths and labels in two lists.
    """
    # certain files and directory that aren't loaded
    skip = ['.DS_Store', '.DS_S_i.png',
            'datasets/matura_data/merge', 'datasets/matura_data/PhaseContrast', 'Session']

    files = []
    labels = []

    # walk through all files in a directory
    for dirpath, dirnames, filenames in os.walk(data_dir):

        for filename in filenames:
            # get path of file
            path = os.path.join(dirpath, filename)

            # ignore files that don't exist, are in the skip list, or who's directory is in the skip list
            if (not os.path.isfile(path) or filename in skip or any([ski in dirpath for ski in skip])) is False:

                # parse label
                label = "not infected" if 'ni' in filename else "infected"
                labels.append(label)

                files.append(os.path.join(dirpath, filename))

    # df = pd.DataFrame(list(zip(files, labels)), columns=["filenames", "labels"])
    return files, labels


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


replace_filetype("datasets/Laloli_et_all2022_wbns_png", "tif", "png")

if __name__ == "__main__":
    main()

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
from skimage.io import imread, imsave
from skimage import exposure

IMG_HEIGHT, IMG_WIDTH = 128, 128

def main():
    commands = {"-help" : None, 
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
            parse_files(input_dir, output_dir, filetype=sys.argv[4])
        else:
            parse_files(input_dir, output_dir)

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
  img = tf.image.resize(img, size = [img_shape, img_shape])
  img = tf.expand_dims(img, axis=0)

  return img


def load_data(data_dir):
    """
    This function loads data from a data directory and their corresponding classficications
    from a given text file.
    The function the returns both lists in a tuple.
    """
    # certain files and directory that aren't loaded
    skip = ['.DS_Store', '.DS_S_i.png', 'Session',
            'datasets/matura_data/merge', 'datasets/matura_data/PhaseContrast'] # 'datasets/Laloli_et_all2022_raw_images',
    
    images = []
    labels = []

    # walk through all files in a directory
    for dirpath, dirnames, filenames in os.walk(data_dir):
    
        for filename in filenames:
            # get path of file
            path = os.path.join(dirpath, filename)

            # ignore files that don't exist, are in the skip list, or who's directory is in the skip list
            if (not os.path.isfile(path) or filename in skip or any([ski in dirpath for ski in skip])) is False:  

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

def load_data_old(data_dir, classficiations):
    """
    This function loads data from a data directory and their corresponding classficications
    from a given text file.
    The function the returns both lists in a tuple.
    """
    images = []
    labels = []

    with open(os.path.join(classficiations), 'r') as f:
        for line in f:

            # parse filename and path
            data = line.split(';')
            filename = data[0]
            path = os.path.join(data_dir, filename)

            if os.path.isfile(path):
                print("parsing line ", line)
            
                # parse label
                label = data[-1]
                labels.append(label)

                # parse image as ndarray
                im = cv2.imread(path)

                # resize image
                resizeIM = cv2.resize(im, (IMG_HEIGHT, IMG_WIDTH))
                print(resizeIM.shape)
                images.append(resizeIM)

    return (images, labels)

def load_data_df(data_dir):
    """
    This function loads
    """
    # certain files and directory that aren't loaded
    skip = ['.DS_Store', '.DS_S_i.png',
            'datasets/matura_data/merge', 'datasets/matura_data/PhaseContrast']
    
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


def parse_files(input_dir, output_dir, filetype="png"):
    """
    This function was used to parse all files from the 'Laloli_et_all2022_raw_images' dir
    that belong to the "GFP"-category. 
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
                image_dark = exposure.adjust_gamma(image, gamma=2,gain=1)
                imsave(dst, image_dark)


def parse_files_old(input_dir, output_dir, filetype="png"):
    """
    This function was used to parse all images from the matura dataset into
    distinct categories (merge, gfp, phase contrast).
    """
    for filename in os.listdir(input_dir):
        if filename in [".DS_Store"]:
            continue
        src = os.path.join(input_dir, filename)

        if "merge" in filename:
            dst = os.path.join(output_dir, "merge", filename.split('.')[0] + '.' + filetype)
        
        elif "GFP" in filename:
            dst = os.path.join(output_dir, "GFP", filename.split('.')[0] + '.' + filetype)

        elif "PhaseContrast" in filename:
            dst = os.path.join(output_dir, "PhaseContrast", filename.split('.')[0] + '.' + filetype)
        
        im = Image.open(src)
        im.save(dst)


def adapt_filenames(input_dir):
    """
    This function was used to change filenames of Laloli dataset images s.t.
    it reflects the infection state.
    """
    for dirpath, dirnames, filenames in os.walk(input_dir):
        for filename in filenames:
            old = os.path.join(dirpath, filename)
            if 'M' in filename:
                print(filename[:-4] + '_ni.png')
                new = os.path.join(dirpath, filename[:-4] + '_ni.png')
            else:
                print(filename[:-4] + '_i.png')
                new = os.path.join(dirpath, filename[:-4] + '_i.png')
            
            os.rename(old, new)

def adapt_filenames_2(input_dir, classifications):
    """
    This function was used to change filenames of matura dataset images s.t.
    it reflects the infection state.
    """
    with open(classifications, 'r') as f:
        for line in f:
            # separating parsed line
            data = line.split(';')
            
            # parse filename
            filename = data[0]
            print(filename)
            label = int(data[-1])

            suffix = '_i.png' if label == 1 else '_ni.png'

            old = os.path.join(input_dir, filename)
            new = os.path.join(input_dir, filename[:-4] + suffix)

            try:
                os.rename(old, new)
            except:
                continue

def adapt_filenames_3(input_dir, filetype="png"):
    infected = os.listdir("datasets/Laloli_et_all2022_png/infected")
    not_infected = os.listdir("datasets/Laloli_et_all2022_png/not_infected")

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

                if 'M' not in filename:
                    print(f"{filename} infected")
                    dst = os.path.join(input_dir, "infected", filename)
                else:
                    print(f"{filename} not infected")
                    dst = os.path.join(input_dir, "not_infected", filename)
                
                shutil.move(src, dst)


adapt_filenames_3("datasets/Laloli_et_all2022_raw_images")


if __name__ == "__main__":
    main()
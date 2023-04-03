"""
evalute.py
This Python module provides functions for evaluating fluorescence images of cell cultures
using a convolutional neural network. Based on the classification (infected / not-infected),
the Tissue-Culture-Infective-Dose-50% (TCID50) can be calculated.
"""

from datetime import datetime
import os
import sys
import numpy as np
import copy
import argparse
from tqdm import tqdm
from pathvalidate.argparse import validate_filepath_arg

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa
print("importing tensorflow modules...")  # noqa
import tensorflow as tf

from data_pipe import load_and_prep_image
from control import get_outlier_rows

IMG_HEIGHT, IMG_WIDTH = (256, 256)


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate a single file or all files in a dir."
    )

    # get function argument
    parser.add_argument("-f", "--function", type=str,
                        help="specify desired functionality of module", required=True, choices=["evaluate_image", "evaluate_dir"])

    # one of those arguments (file path to image or directory path) is required
    parser.add_argument("-p", "--image_path", type=validate_filepath_arg,
                        required="evaluate_image" in sys.argv)

    parser.add_argument("-d", "--image_dir", type=check_dir_path,
                        required="evaluate_dir" in sys.argv)

    # parse path to model
    parser.add_argument("-m", "--model", type=check_dir_path, required=True)

    args = vars(parser.parse_args())

    print("loading model...")
    model = tf.keras.models.load_model(args["model"])

    if args["function"] == "evaluate_image":

        # get prediction
        state = evaluate(image_path=args["image_path"], model=model,
                         classnames=["infected", "not infected"])

        print(state[0], " with ", round(state[1] * 100, 2), "% confidence")

    else:
        sub = int(input("How many subdivisions per plate?: "))
        row_ranges = []
        col_ranges = []

        for i in range(sub):
            # ask for user input to specify plate coordinates
            row_range = [input("Enter first row (single letter): "),
                         input("Enter last row (single letter): ")]

            col_range = [int(input("Enter first column (integer): ")),
                         int(input("Enter last column (integer): "))]

            row_ranges.append(row_range)
            col_ranges.append(col_range)

        evaluate_plates(args["image_dir"], row_ranges, col_ranges, model)


def check_dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(
            f"readable_dir:{path} is not a valid path")


def evaluate(image_path, model=None, classnames=[1, 0]):
    """
    This function evaluates (i.e. predicts) the infection state of an image with
    specified by 'image_path' and using a certain model and returns the most likely 
    prediction class.

    image_path: string specifying path to image.
    model: A TensorFlow Keras model to be used for prediction.
    classnames: A list of class names. By default, this is set to [1, 0] to represent 
    "infected" and "not infected" respectively.

    It returns a tuple containing the predicted class name and the maximum prediction value.
    """
    # load image and get a prediction
    prediction = model.predict(load_and_prep_image(image_path), verbose=0)

    if len(prediction[0]) > 1:
        # return most likely class of prediction
        return classnames[int(tf.round(prediction)[0][0])], max(prediction[0])

    else:
        return int(prediction[0][0] >= 0.5), prediction[0][0]


def evaluate_plate(plate, model):
    """
    This function takes a matrix of paths to images as an argument,
    evaluates each image with a specified model and returns the
    predictions.

    plate: A matrix representing the plate of cell cultures.
    model: A TensorFlow Keras model to be used for prediction.
    """

    outputs = []

    for row in plate:
        eval_row = []
        for file in row:
            prediction = evaluate(file, model)
            res = prediction[0]
            eval_row.append(res)
        outputs.append(eval_row)

    return outputs


def checked_plate(plate, eval_plate, classfile):
    """
    This function returns the actual (i.e., human labeled) values for a
    cell culture image.
    """
    # remove late
    files = open(classfile, 'r')
    check_dict = {
        f.split(";")[0]: int(f.split(";")[1].replace("\n", ""))
        for f in files
    }

    check_plate = []
    corr_plate = copy.deepcopy(eval_plate)
    for i in range(len(plate)):
        row = []
        for j in range(len(plate[i])):
            row.append(check_dict[plate[i][j]] == eval_plate[i][j])
            corr_plate[i][j] = int(check_dict[plate[i][j]])
        check_plate.append(row)

    delta_titer = np.log10(spear_karb(corr_plate, 1, -1)) - \
        np.log10(spear_karb(eval_plate, 1, -1))
    return check_plate, delta_titer


def get_plates(data_dir, row_range, col_range):
    """
    This function returns a matrix containing file paths to image files 
    from a directory containing file paths to image files. 
    It accepts three arguments:

    data_dir: A string representing the path to the directory containing the image files.
    col_range: A tuple representing the range of columns on the plate.
    row_range: A tuple representing the range of rows on the plate.
    """
    filenames = os.listdir(data_dir)
    plate = []

    row_range = [chr(i)
                 for i in range(ord(row_range[0]), ord(row_range[1]) + 1)]
    col_range = list(range(col_range[0], col_range[1] + 1))

    # loop through all coordinates of a plate
    for row_nb in row_range:
        row = []

        for col_nb in col_range:
            coord = row_nb + str(col_nb)

            # find matches
            matches = [file for file in filenames if coord in file]
            file_path = os.path.join(data_dir, matches[0])
            row.append(file_path)

        plate.append(row)

    return plate


def evaluate_plates(data_dir, row_ranges, col_ranges, model):
    """
    This function evaluates all plates that are located in data_dir.
    """
    # create saving directory
    idx = len(os.listdir("plots/evaluated_plates"))-1
    save_dir = f"plots/evaluated_plates/run_{idx}/"
    os.mkdir(save_dir)

    # correction of titers
    corrs = []
    plate_dirs = [dir for dir in os.listdir(data_dir)
                  if os.path.isdir(os.path.join(data_dir, dir))]

    for plate_path in tqdm(plate_dirs, total=len(plate_dirs)):

        for row_range, col_range in zip(row_ranges, col_ranges):
            filepath = os.path.join(data_dir, plate_path)

            plate = get_plates(filepath, row_range, col_range)

            eval_plate = evaluate_plate(plate, model)
            check_plate, corr = checked_plate(
                plate, eval_plate, "classifications/classification.txt")
            corrs.append(corr)

            titer = spear_karb(eval_plate, 1, -1)
            outlier_rows = get_outlier_rows(eval_plate, titer, 10, 10, 35)

            # display_plate(eval_plate)
            rows = "".join(row_range)
            cols = "".join([str(i) for i in col_range])

            save_checked_plate(eval_plate, check_plate, titer, corr, outlier_rows, row_range,
                               col_range, save_path=f"{save_dir}/{plate_path}{rows}_{cols}{'%.2E' % titer}.png")
            # print('%.2E' % titer)

    print(
        f"\n\n End result: mean difference {np.mean(corrs)} ± {np.std(corrs)}")


def display_plate(plate):
    """
    This function creates a terminal representation of a well plate where
    each infected well is marked with an X. It accepts
    one argument:

    plate: A matrix representing the plate of cell cultures.
    """
    n = len(plate[0])
    print("\n\n" + n * "----")
    for row in plate:
        print('|', end="")
        for well in row:
            val = " X" if well == 1 else "  "
            print(val, end="")
            print(" |", end="")
        print("\n" + n * "----")
    print("\n\n")


def save_checked_plate(plate, checked_plate, titer, delta_titer, outlier_rows, row_range, col_range, save_path):
    row_range = [chr(i)
                 for i in range(ord(row_range[0]), ord(row_range[1]) + 1)]
    col_range = list(range(col_range[0], col_range[1] + 1))

    well_size = 100
    well_border = 10

    nb_col = len(plate[0])
    nb_row = len(plate)

    # Create a blank canvas
    img = Image.new("RGBA",
                    ((nb_col + 5) * well_size,
                     (nb_row + 1) * well_size)
                    )

    font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
    draw = ImageDraw.Draw(img)

    for i in range(1, nb_row + 1):
        # color is red if outlier, else white
        outlier = "    *" if (i-1) in outlier_rows else ""
        draw.text((nb_col+1, i*well_size), outlier, fill="black", font=font)

        # draw row letter
        draw.text((0, i * well_size),
                  row_range[i-1], fill="black", font=font)

        draw.text(((nb_col+1)*well_size, i*well_size),
                  outlier, fill="black", font=font)

        for j in range(1, nb_col + 1):
            # draw column number
            draw.text((j * well_size + 2 * well_border, 0),
                      str(col_range[j-1]), fill="black", font=font)

            # define dimensions of well
            dims = ((j * well_size + well_border,
                     i * well_size + well_border),
                    ((j + 1) * well_size - well_border,
                     (i + 1) * well_size - well_border)
                    )

            # draw a well
            color = "white" if checked_plate[i-1][j-1] else "red"
            draw.rectangle(dims, fill=color, outline="black", width=3)

            # if infected, draw a cross
            if plate[i-1][j-1]:
                cross = (dims,
                         ((j * well_size + well_border, (i + 1) * well_size - well_border),
                          ((j + 1) * well_size - well_border, i * well_size + well_border))
                         )

                draw.line(cross[0], fill="black", width=3)
                draw.line(cross[1], fill="black", width=3)

    # draw titer
    draw.text((((nb_row) * well_size / 2),
               (nb_col + 3) * well_size),
              f"TCID50={titer}\n \delta logtiter={delta_titer}",
              fill="black",
              font=font
              )

    img.save(save_path)


def save_plate(plate, titer, outlier_rows, row_range, col_range, save_path):
    """
    This function creates an illustrative png of a well plate an marks
    each infected well with a cross.
    """
    row_range = [chr(i)
                 for i in range(ord(row_range[0]), ord(row_range[1]) + 1)]
    col_range = list(range(col_range[0], col_range[1] + 1))

    well_size = 100
    well_border = 10

    nb_col = len(plate[0])
    nb_row = len(plate)

    # Create a blank canvas
    img = Image.new("RGBA",
                    ((nb_col + 3) * well_size,
                     (nb_row + 1) * well_size)
                    )

    font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
    draw = ImageDraw.Draw(img)

    for i in range(1, nb_row + 1):
        # color is red if outlier, else white
        color = "red" if (i-1) in outlier_rows else "white"

        # draw row letter
        draw.text((0, i * well_size),
                  row_range[i-1], fill="black", font=font)

        for j in range(1, nb_col + 1):
            # draw column number
            draw.text((j * well_size + 2 * well_border, 0),
                      str(col_range[j-1]), fill="black", font=font)

            # define dimensions of well
            dims = ((j * well_size + well_border,
                     i * well_size + well_border),
                    ((j + 1) * well_size - well_border,
                     (i + 1) * well_size - well_border)
                    )

            # # purple if outlier
            # color = "purple" if plate[i-1][j-1] == 0.5 else "white"

            # draw a well
            draw.rectangle(dims, fill=color, outline="black", width=3)

            # if infected, draw a cross
            if plate[i-1][j-1]:
                cross = (dims,
                         ((j * well_size + well_border, (i + 1) * well_size - well_border),
                          ((j + 1) * well_size - well_border, i * well_size + well_border))
                         )

                draw.line(cross[0], fill="black", width=3)
                draw.line(cross[1], fill="black", width=3)

    # draw titer
    draw.text(((nb_col + 3) * well_size / 2,
               ((nb_row + 1) * well_size)),
              f"TCID50={titer}",
              fill="black",
              font=font
              )

    img.save(save_path)


def spear_karb(plate, d, d_0):
    """
    This function returns the Tissue-Culture-Infective-Dose-50% (TCID50) value based
    on a plate that is represented as a matrix. Each column and row corresponds to
    the position of the cell culture well on the plate, and the entry (either 0 or 1)
    reflects the infection state. The function calculates the log10 concentration of
    the first fully infected row and returns the TCID50 value. It accepts three
    arguments:

    plate: A matrix representing the plate of cell cultures.
    d: A float representing the log10 of the dilution factor.
    d_0: A float representing the log10 of the dilution of the first well.
    """
    # special case: no fully infected rows
    if not any([sum(row) == len(row) for row in plate]):
        d_0 += 0.5

    # find fully infected row with greatest dilution
    row0 = most_diluted(plate)

    # calculate the log10 concentration of the first fully infected row
    x_0 = d_0 - (row0) * d

    # calculate sum of fractions of infected wells per row
    s = 0

    # smooth out data
    plate = sorted(plate, key=lambda row: (sum(row) / len(row)), reverse=True)
    # remove duplicates
    p = []
    [p.append(row) for row in plate if row not in p]

    for row in p:
        s += (sum(row) / len(row))

    return 10 ** -((x_0 + d/2 - d * s) + d_0)


def most_diluted(plate):
    """
    This function finds the most diluted row in the plate matrix
    that's still fullyinfected and returns the index of that row 
    as an int.

    plate: A matrix representing the plate of cell cultures.
    """
    row0 = 0
    for row in range(len(plate)):
        if sum(plate[row]) == len(plate[row]):
            row0 = row
    return row0


if __name__ == "__main__":
    main()

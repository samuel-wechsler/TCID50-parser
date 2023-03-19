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

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

from data_pipe import load_and_prep_image

IMG_HEIGHT, IMG_WIDTH = (256, 256)


def main():
    """
    This function is the entry point of the program. It processes command
    line arguments and calls the appropriate function(s) based on the
    commands provided.
    """

    commands = {"-help": None, "-evaluate": ("file", "model")}

    if sys.argv[1] not in commands:
        sys.exit("not a valid command: python evaluate.py -help")

    if "-help" in sys.argv:
        print("See a list of all commands:")
        for com in commands.keys():
            if commands[com] is not None:
                print(com, " ".join(list(commands[com])))
            else:
                print(com)

    elif "-evaluate" in sys.argv:
        if len(sys.argv) != 4:
            sys.exit("not a valid command: python evaluate.py -help")

        src = sys.argv[2]
        model = sys.argv[3]

        if not (os.path.isfile(src)):
            sys.exit("not a valid directory")

        print("loading model...")
        model = tf.keras.models.load_model(model)

        state = evaluate(src, model, classnames=["infected", "not infected"])
        print(state[0], " with ", round(state[1] * 100, 2), "% confidence")


def evaluate(dir, model, classnames=[1, 0]):
    """
    This function evaluates (i.e. predicts) the infection state of an image with
    a directory 'dir' and using a certain model and returns the most likely 
    prediction class.

    dir: A string representing the path to the directory containing the image file(s).
    model: A TensorFlow Keras model to be used for prediction.
    classnames: A list of class names. By default, this is set to [1, 0] to represent 
    "infected" and "not infected" respectively.

    It returns a tuple containing the predicted class name and the maximum prediction value.
    """
    # load image and get a prediction
    prediction = model.predict(load_and_prep_image(dir))

    # return most likely class of prediction
    return classnames[int(tf.round(prediction)[0][0])], max(prediction[0])


def evaluate_plate(plate, model):
    """
    This function takes a matrix of paths to images as an argument,
    evaluates each image with a specified model and returns the
    predictions.

    plate: A matrix representing the plate of cell cultures.
    model: A TensorFlow Keras model to be used for prediction.
    """

    eval_plate = []

    for row in plate:
        eval_row = []
        for file in row:
            res = evaluate(file, model)[0]
            eval_row.append(res)
        eval_plate.append(eval_row)

    return eval_plate


def get_plates(data_dir, col_range, row_range):
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


def save_plate(plate, row_range, col_range, save_dir):
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
                    ((nb_col + 1) * well_size,
                     (nb_row + 1) * well_size)
                    )

    font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
    draw = ImageDraw.Draw(img)

    for i in range(1, nb_row + 1):
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

            # draw a well
            draw.rectangle(dims, fill="white", outline="black", width=3)

            # if infected, draw a cross
            if plate[i-1][j-1]:
                cross = (dims,
                         ((j * well_size + well_border, (i + 1) * well_size - well_border),
                          ((j + 1) * well_size - well_border, i * well_size + well_border))
                         )

                draw.line(cross[0], fill="black", width=3)
                draw.line(cross[1], fill="black", width=3)

    img.save(save_dir)


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
    row0 = find_x0(plate)

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


def find_x0(plate):
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


plate = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
]

# plate = [
#     [1, 1, 1, 1],
#     [1, 1, 1, 1],
#     [1, 1, 1, 1],
#     [1, 1, 1, 1],
#     [1, 1, 1, 1],
#     [1, 1, 1, 1],
#     [1, 1, 1, 1],
#     [1, 1, 1, 0],
#     [1, 0, 0, 0],
#     [0, 0, 0, 0],
# ]

# # res = spear_karb(plate, 1, -1)

# t1 = datetime.now()

# model = tf.keras.models.load_model("trained_models/raw_model")
# bp = "/Volumes/T7/tcid50_datasets/Lukas_Probst_IDV/Lukas_Probst_IDV_ori/Titration Timecourse"

# plates = [plate for plate in os.listdir(bp) if 'docx' not in plate]
# plates = sorted(plates, key=lambda x: int(x[-2:]))

# for plate in plates:
#     plate = plate.replace("._", "")

#     print("\n\n ", plate)

#     plate_1 = get_plates(os.path.join(bp, plate), [1, 6], ["A", "H"])
#     plate_2 = get_plates(os.path.join(bp, plate), [7, 12], ["A", "H"])

#     plate_1 = evaluate_plate(plate_1, model)
#     plate_2 = evaluate_plate(plate_2, model)

#     titer_1 = spear_karb(plate_1, 1, -1)
#     titer_2 = spear_karb(plate_2, 1, -1)

#     display_plate(plate_1)
#     print('%.2E' % titer_1)

#     display_plate(plate_2)
#     print('%.2E' % titer_2)

# t2 = datetime.now()

# print("Finished in ", t2 - t1)

if __name__ == "__main__":
    main()

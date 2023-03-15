"""
evalute.py

This python module is used to evaluate fluorescence images of cell cultures using a convolutional
neural network. Based on the classification (infected / not-infected) the Tissue-Culture-Infective-
Dose-50% (TCID50) can be calculated.
"""

import os
import sys
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf

from data_pipe import load_and_prep_image

IMG_HEIGHT, IMG_WIDTH = (256, 256)

def main():
    commands = {"-help":None, "-evaluate": ("file", "model")}

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

        if not(os.path.isfile(src)):
            sys.exit("not a valid directory")
        
        print("loading model...")
        model = tf.keras.models.load_model(model)

        state = evaluate(src, model, classnames=["infected", "not infected"])
        print(state[0], " with ", round(state[1] * 100, 2), "% confidence")


def evaluate(dir, model, classnames=[1,0]):
    """
    This function evaluates (i.e. predicts) the infection state of an image with
    a directory 'dir' and using a certain model and returns the most likely 
    prediction class.
    """
    # load image and get a prediction
    prediction = model.predict(load_and_prep_image(dir))
    
    # return most likely class of prediction
    return classnames[int(tf.round(prediction)[0][0])], max(prediction[0])


def evaluate_plate(plate, data_dir):
    model = tf.keras.models.load_model("gfp_model")
    eval_plate = []

    for row in plate:
        eval_row =[]
        for file in row:
            # parse image as ndarray
            im = load_and_prep_image(os.path.join(data_dir, file))
            res = model.predict(im)
            eval_row.append(res)
            
    return eval_plate

def spear_karb(plate, d, d_0):
    """
    This function returns as TCDI50 value based on a plate that is represented as a matrix.
    Each column and row corresponds to the position of the cell culture well on the plate, the
    entry (either 0 or 1) reflects the infection state.
    d: log10 of dilution factor
    d_0: log10 of dilution of the first well
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
    """
    row0 = 0
    for row in range(len(plate)):
        if sum(plate[row]) == len(plate[row]):
            row0 = row
    return row0
        

# plate = [
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 1, 1, 0],
#     [1, 1, 1, 1, 1, 1, 0, 0],
#     [1, 1, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0]
# ]

plate = [
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 0],
]

# res = spear_karb(plate, 1, -1)

# print('%.2E' % res)

if __name__ == "__main__":
    main()
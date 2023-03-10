"""
evalute.py

This python module is used to evaluate fluorescence images of cell cultures using a convolutional
neural network. Based on the classification (infected / not-infected) the Tissue-Culture-Infective-
Dose-50% (TCID50) can be calculated.
"""

import os
import numpy as np

import tensorflow as tf

IMG_WIDTH = 256
IMG_HEIGHT = 256

# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=256):
  """
  Reads an image from filename, turns it into a tensor
  and reshapes it to (img_shape, img_shape, colour_channel).
  """
  # Read in target file (an image)
  img = tf.io.read_file(filename)

  # Decode the read file into a tensor & ensure 3 colour channels 
  # (our model is trained on images with 3 colour channels and sometimes images have 4 colour channels)
  img = tf.image.decode_image(img, channels=3)

  # Resize the image (to the same size our model was trained on)
  img = tf.image.resize(img, size = [img_shape, img_shape])
  img = tf.expand_dims(img, axis=0)

  return img

class_names = ["infected", "not infected"]
model = tf.keras.models.load_model("gfp_model")
pred = model.predict(load_and_prep_image("data/GFP/Hep2_day7_E4_GFP.png"))
pred_class = class_names[int(tf.round(pred)[0][0])]
print(pred_class)

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
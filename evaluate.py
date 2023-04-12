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
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa
print("importing tensorflow modules...")  # noqa
import tensorflow as tf

from data_pipe import load_and_prep_image
from control import get_outlier_rows

IMG_HEIGHT, IMG_WIDTH = (128, 128)


def main():

    parser = argparse.ArgumentParser(
        description="-"
    )

    # get function argument
    parser.add_argument("-f", "--function", type=str,
                        help="specify desired functionality of module", required=True, choices=["evaluate_image", "evaluate_dir"])

    # one of those arguments (file path to image or directory path) is required
    parser.add_argument("-p", "--image_path", type=validate_filepath_arg,
                        required="evaluate_image" in sys.argv)

    parser.add_argument("-d", "--image_dir", type=check_dir_path,
                        required="evaluate_dir" in sys.argv)

    parser.add_argument("-c", "--check_file", type=ValueError)

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
            row_range = [ask_row_range("first"), ask_row_range("second")]

            col_range = [ask_col_range("first"), ask_col_range("second")]

            row_ranges.append(row_range)
            col_ranges.append(col_range)

        print(f"\nstarting titration readout...\n")

        if args["check_file"] is None:
            eval = EvaluatePlates(
                args["image_dir"], row_ranges, col_ranges, model)
        else:
            eval = EvaluateCheckPlates(args["image_dir"], args["check_file"],
                                       row_ranges, col_ranges, model)
            mn, std = eval.get_diffs()
            print(f"\nmean error: {mn} ± {std}")


def check_dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(
            f"readable_dir:{path} is not a valid path")


def ask_row_range(idx):
    is_char, is_upper = False, False

    while not (is_char and is_upper):
        char = input(f"Enter {idx} row (single letter): ")

        if type(char) == str:
            is_char = char.isalpha() and len(char) == 1
            is_upper = char.isupper()

    return char


def ask_col_range(idx):
    intg = ""

    while not intg.isalnum():
        intg = input(f"Enter {idx} column (integer): ")
    return int(intg)


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
        return int(prediction[0][0] >= 0.5), 2*(max(prediction[0][0], 1 - prediction[0][0]) - 0.5)


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


class EvaluateUtils(object):
    """ Utils for evaluation of plates """

    def create_run_dir(self, base_dir):
        id = len(os.listdir(base_dir)) - 1
        self.save_dir = os.path.join(base_dir, f"run_{id}")

        while os.path.isdir(self.save_dir):
            id += 1
            self.save_dir = os.path.join(base_dir, f"run_{id}")
        os.mkdir(self.save_dir)

    def get_image_paths(self, dir, row_range, col_range):
        """
        This function returns a matrix containing file paths to image files 
        from a directory containing file paths to image files. 
        It accepts three arguments:

        data_dir: A string representing the path to the directory containing the image files.
        col_range: A tuple representing the range of columns on the plate.
        row_range: A tuple representing the range of rows on the plate.
        """
        filenames = os.listdir(dir)
        plate = []

        row_range = [chr(i) for i in range(
            ord(row_range[0]), ord(row_range[1]) + 1)]

        col_range = list(range(col_range[0], col_range[1] + 1))

        # loop through all coordinates of a plate
        for row_nb in row_range:
            row = []

            for col_nb in col_range:
                coord = row_nb + str(col_nb)

                # find matches
                matches = [file for file in filenames if coord in file]
                file_path = os.path.join(dir, matches[0])
                row.append(file_path)

            plate.append(row)

        return plate

    def evaluate_plates(self):
        """
        Adapted version of evaluate_plates
        """
        for plate_dir in tqdm(self.plate_dirs, total=len(self.plate_dirs)):

            for row_range, col_range in zip(self.row_ranges, self.col_ranges):
                plate_path = os.path.join(self.data_dir, plate_dir)

                # get matrix of file names of a plate
                plate = self.get_image_paths(plate_path, row_range, col_range)

                # evaluate plate, get titer and outliers
                evaluated_plate = self.evaluate_plate(plate)
                titer = spear_karb(evaluated_plate, 1, -1)
                outliers = get_outlier_rows(evaluated_plate, titer, 10, 10, 35)

                # append to manual_recheck list if outliers present
                if len(outliers) > 0:
                    for i in outliers:
                        evaluated_plate[i] = plate[i]
                        self.manual_checks.append(
                            (plate_dir, plate, evaluated_plate))

                else:
                    rows = "".join(row_range)
                    cols = "".join([str(i) for i in col_range])

                    self.save_plate(evaluated_plate, titer, outliers, row_range,
                                    col_range, save_path=f"{self.save_dir}/{plate_dir}{rows}_{cols}{'%.2E' % titer}.png")

    def evaluate_plate(self, plate):
        """
        This method takes a matrix of paths to images as an argument, evaluates each image with a specified model
        and returns the predictions.
        """

        outputs = []

        for row in plate:
            eval_row = []
            for file in row:
                prediction = evaluate(file, self.model)
                res = prediction[0]
                confidence = prediction[1]
                eval_row.append(res)
            outputs.append(eval_row)

        return outputs

    def user_check(self):
        """
        Asking user for manual control in case of outlier rows
        """
        for plate_dir, plate, evaluated_plate in self.manual_checks:
            for row_range, col_range in zip(self.row_ranges, self.col_ranges):
                for i in range(len(evaluated_plate)):
                    if any([os.path.isfile(p) for p in evaluated_plate[i]]):
                        evaluated_plate[i] = self.ask_user(evaluated_plate[i])

                titer = spear_karb(evaluated_plate, 1, -1)

                # get outliers in plate
                outlier_rows = get_outlier_rows(
                    evaluated_plate, titer, 10, 10, 35)
                # save graphic representation of plate
                rows = "".join(row_range)
                cols = "".join([str(i) for i in col_range])

                self.save_plate(evaluated_plate, titer, outlier_rows, row_range, col_range,
                                save_path=f"{self.save_dir}/{plate_dir}_{rows}_{cols}{'%.2E' % titer}.png")

    def ask_user(self, row):
        """ helper method for user_check """
        evals = []
        for path in row:
            filename = os.path.splitext(os.path.split(path)[0])[0]
            img = cv2.imread(path)

            cv2.imshow("image", img)
            cv2.waitKey(0)
            # closing all open windows
            cv2.destroyAllWindows()

            classification = input(f"{filename} infected (0 / 1): ")
            while classification not in ["0", "1"]:
                classification = input(f"{filename} infected (0 / 1): ")

            evals.append(int(classification))

        return evals

    def save_plate(self, plate, titer, outlier_rows, row_range, col_range, save_path):
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
            outlier = "    *" if (i-1) in outlier_rows else ""

            # draw row letter
            draw.text((0, i * well_size),
                      row_range[i-1], fill="black", font=font)

            # mark potential outliers
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

                # # purple if outlier
                # color = "purple" if plate[i-1][j-1] == 0.5 else "white"

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

        # draw titer
        draw.text((((nb_row) * well_size / 2),
                   (nb_col + 3) * well_size),
                  f"TCID50={titer}\n",
                  fill="black",
                  font=font
                  )

        img.save(save_path)

    def display_plate(self, plate):
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


class EvaluatePlates(EvaluateUtils):
    """ class to evaluate all plates in a directory """

    def __init__(self, data_dir, row_ranges, col_ranges, model):
        self.data_dir = data_dir
        self.row_ranges = row_ranges
        self.col_ranges = col_ranges
        self.model = model

        # initiate list variables
        self.plate_dirs = [dir for dir in os.listdir(data_dir)
                           if os.path.isdir(os.path.join(data_dir, dir))]
        self.manual_checks = []

        # start evaluation process
        self.create_run_dir("plots/evaluated_plates")
        self.evaluate_plates()
        self.user_check()


class EvaluateCheckPlates(EvaluateUtils):

    def __init__(self, data_dir, check_file_path, row_ranges, col_ranges, model):
        self.data_dir = data_dir
        self.check_file_path = check_file_path
        self.row_ranges = row_ranges
        self.col_ranges = col_ranges
        self.model = model

        # initiate list variables
        self.plate_dirs = [dir for dir in os.listdir(data_dir)
                           if os.path.isdir(os.path.join(data_dir, dir))]
        self.diffs = []
        self.manual_checks = []

        # start evaluation process
        self.create_run_dir("plots/evaluated_plates")
        self.evaluate_plates()
        self.user_check()

    def evaluate_plates(self):
        """
        Adapted version of evaluate_plates
        """
        for plate_dir in tqdm(self.plate_dirs, total=len(self.plate_dirs)):

            for row_range, col_range in zip(self.row_ranges, self.col_ranges):
                plate_path = os.path.join(self.data_dir, plate_dir)

                # get matrix of file names of a plate
                plate = self.get_image_paths(plate_path, row_range, col_range)

                # evaluate plate, get titer and outliers
                evaluated_plate = self.evaluate_plate(plate)
                titer = spear_karb(evaluated_plate, 1, -1)
                outliers = get_outlier_rows(evaluated_plate, titer, 10, 10, 35)

                # case: outliers
                if len(outliers) > 0:
                    for i in outliers:
                        # indicated which rows to recheck manually
                        evaluated_plate[i] = plate[i]
                        self.manual_checks.append((plate_dir, plate, evaluated_plate,
                                                   row_range, col_range))

                # case: no outliers
                else:
                    # check evaluation of plate with human labeled data
                    check_plate, diff = self.check_plate(plate,
                                                         evaluated_plate)
                    # store difference of calculated vs actual titer
                    self.diffs.append(diff)

                    rows = "".join(row_range)
                    cols = "".join([str(i) for i in col_range])

                    # save graphical representation of evaluated plate
                    self.save_plate(evaluated_plate, check_plate, titer, diff, outliers, row_range,
                                    col_range, save_path=f"{self.save_dir}/{plate_dir}_{rows}_{cols}.png")

    def check_plate(self, plate, eval_plate):
        """
        This function returns the actual (i.e., human labeled) values for a
        cell culture image.
        """
        # remove late
        files = open("classifications/classification.txt", 'r')

        # get human labels
        labels = {
            f.split(";")[0]: int(f.split(";")[1].replace("\n", ""))
            for f in files
        }

        checked_plate = []
        corr_plate = copy.deepcopy(eval_plate)

        for i in range(len(plate)):
            row = []
            for j in range(len(plate[i])):
                row.append(labels[plate[i][j]] == eval_plate[i][j])
                corr_plate[i][j] = int(labels[plate[i][j]])
            checked_plate.append(row)

        delta_titer = abs(np.log10(spear_karb(corr_plate, 1, -1)) -
                          np.log10(spear_karb(eval_plate, 1, -1)))

        return checked_plate, delta_titer

    def user_check(self):
        """
        Asking user for manual control in case of outlier rows
        """
        for plate_dir, plate, evaluated_plate, row_range, col_range in self.manual_checks:
            # ask user in case there are file paths left in row
            for i in range(len(evaluated_plate)):
                if any([os.path.isfile(p) for p in evaluated_plate[i]]):
                    evaluated_plate[i] = self.ask_user(evaluated_plate[i])

            titer = spear_karb(evaluated_plate, 1, -1)

            # get outliers in plate
            outlier_rows = get_outlier_rows(
                evaluated_plate, titer, 10, 10, 35)
            check_plate, diff = self.check_plate(plate, evaluated_plate)

            self.diffs.append(diff)

            # save graphic representation of plate
            rows = "".join(row_range)
            cols = "".join([str(i) for i in col_range])

            self.save_plate(evaluated_plate, check_plate, titer, diff, outlier_rows, row_range, col_range,
                            save_path=f"{self.save_dir}/{plate_dir}_{rows}_{cols}.png")

    def save_plate(self, plate, checked_plate, titer, delta_titer, outlier_rows, row_range, col_range, save_path):
        row_range = [chr(i) for i in range(
            ord(row_range[0]), ord(row_range[1]) + 1)]
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
            draw.text((nb_col+1, i*well_size),
                      outlier, fill="black", font=font)

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

    def get_diffs(self):
        return np.mean(self.diffs), np.std(self.diffs)


if __name__ == "__main__":
    main()

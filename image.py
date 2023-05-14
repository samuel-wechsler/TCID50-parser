from train_test import get_model
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os


def get_image_paths(dir, row_range, col_range):
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


# imgs = get_image_paths("/Volumes/T7/tcid50_datasets/Lukas_Probst_IDV/Lukas_Probst_IDV_ori/Titration Timecourse/220518_162808_Plate 4",
#                        ["A", "H"], [1, 6])
# img_arrs = []

# for row in imgs:
#     for img in row:
#         img_arrs.append(Image.open(img))

# ax = 8
# ay = 6
# fig = plt.figure(figsize=(18, 24))

# c = 0
# for i in range(8):
#     for j in range(6):
#         sub = fig.add_subplot(ax, ay, c + 1)
#         sub.axis('off')
#         sub.imshow(img_arrs[c], cmap=cm.gray)
#         c += 1
# plt.show()
# fig.savefig("example.png", dpi=500)

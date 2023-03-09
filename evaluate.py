"""
evalute.py

This python module is used to evaluate fluorescence images of cell cultures using a convolutional
neural network. Based on the classification (infected / not-infected) the Tissue-Culture-Infective-
Dose-50% (TCID50) can be calculated.
"""

import os

def parse_plates(day, cell, data_dir):
    coords = []

    for filename in os.listdir(data_dir):
        if (day in filename) and (cell in filename):
            # parse column and row
            row_col = filename.split('_')[2]
            col = row_col[:1]
            row = row_col[1:]
            coords.append((row, col))
    
    return coords

parse_plates("day3", "Hep2", "data/merge")

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

res = spear_karb(plate, 1, -1)

print('%.2E' % res)
import numpy as np
from decimal import Decimal

def evaluate(plate, d_0, d, c_0):
    """
    This function returns as TCDI50 value based on a plate that is represented as a matrix.
    Each column and row corresponds to the position of the cell culture well on the plate, the
    entry (either 0 or 1) reflects the infection state.
    d: log10 of dilution factor
    d_0: log10 of dilution of the first well
    c_0: log10 of lowest dilution
    """
    # find fully infected row with greatest dilution
    row0 = find_x0(plate)
    x_0 = c_0 - (row0) * d

    # calculate sum of fractions of infected wells per row
    s = 0
    for row in plate[row0:]:
        s += (sum(row) / len(row))

    return (x_0 + d/2 - d * s) - d_0


def find_x0(plate):
    """
    This function finds the most diluted row in the plate matrix
    that's still fullyinfected and returns the index of that row 
    as an int.
    """
    row0 = None
    for row in range(len(plate)):
        if sum(plate[row]) == len(plate[row]):
            row0 = row
    return row0
        
plate = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 1, 0],
    [0, 1, 0, 1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0]
]

res = evaluate(plate, 1, 1, -1, )

print('%.2E' % 10 ** (-res))
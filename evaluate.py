"""
evalute.py

This python module is used to evaluate fluorescence images of cell cultures using a convolutional
neural network. Based on the classification (infected / not-infected) the Tissue-Culture-Infective-
Dose-50% (TCID50) can be calculated.
"""


def spear_karb(plate, d_0, d, c_0):
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

    # assert that there is a fully inected row
    if row0 is None:
        raise Exception("There are no fully infected rows. The initial virus titer must have been to low.")
    x_0 = c_0 - (row0) * d

    # calculate sum of fractions of infected wells per row
    s = 0
    for row in plate[row0:]:
        s += (sum(row) / len(row))

    return 10 ** -((x_0 + d/2 - d * s) - d_0)


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
    [1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 0, 1, 1, 0],
    [0, 1, 0, 1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0]
]

res = spear_karb(plate, 1, 1, -1, )

print('%.2E' % res)
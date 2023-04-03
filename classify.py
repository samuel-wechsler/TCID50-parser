"""
classify.py

This python program helps classifying cell images into "infected" or "not-infected"
categories.
"""

import os
import sys
import cv2
from data_pipe import get_file_paths


def main():

    # check command-line arguments
    if len(sys.argv) != 3:
        sys.exit("Usage: python classify.oy data/dir output_file")

    data_dir = sys.argv[1]
    output = open(sys.argv[2], 'a')
    files = open(sys.argv[2], 'r').readlines()
    files = [f.split(';')[0].replace('\n', '') for f in files]

    # walk through a directory
    file_paths = get_file_paths(data_dir)

    for path in file_paths:
        if path in files:
            print("already classified", path)
            continue

        filename, ext = os.path.splitext(path)

        try:
            # show image
            img = cv2.imread(path)
            cv2.imshow("image", img)

            # waits for user to press any key
            # (this is necessary to avoid Python kernel form crashing)
            cv2.waitKey(0)

            # closing all open windows
            cv2.destroyAllWindows()

        except Exception as e:
            print("Error: ", e)
            print("File: ", path)

            ans = input("Continue ? (y/n) : ")
            while ans != "y":
                ans = input("Continue ? (y/n) : ")
            continue

        # ask for classification
        classification = input(f"{filename}: infected? (0 / 1): ")

        # check that input is valid
        while classification not in ["0", "1"]:
            classification = input(f"{filename}: infected? (0 / 1): ")

        # add classification data to txt file
        line = f"{path};{classification}\n"
        output.write(line)


if __name__ == "__main__":
    main()

# File:  /Volumes/T7/tcid50_datasets/fluocells/Mar23bS1C5R3_DMr_200x_y_ll.png

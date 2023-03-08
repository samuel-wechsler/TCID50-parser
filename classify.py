"""
classify.py

This python program helps classifying cell images into "infected" or "not-infected"
categories.
"""

import os
import sys
from PIL import Image

def main():
    
    # check command-line arguments
    if len(sys.argv) != 3:
        sys.exit("Usage: python classify.oy data/dir output_file")

    dst_dir = sys.argv[1]
    output = open(sys.argv[2])

    # loop through all files
    for file in os.listdir(dst_dir):
        data = file.split("_")[:-1]

        # show image
        img = Image.open(os.path.join(dst_dir, file))
        img.show()

        # ask for classification 
        classification = input(f"{file}: infected? (0 / 1): ")

        # check that input is valid
        while classification not in ["0", "1"]:
            classification = input(f"{file}: infected? (0 / 1): ")

        # close file again
        img.close()

        # add classification data to txt file
        data.append(classification)
        output.write(";".join(data) + "\n")


if __name__ == "__main__":
    main()
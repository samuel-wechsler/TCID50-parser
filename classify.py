"""
classify.py

This python program helps classifying cell images into "infected" or "not-infected"
categories.
"""

import os
import sys
import cv2


def main():

    # check command-line arguments
    if len(sys.argv) != 3:
        sys.exit("Usage: python classify.oy data/dir output_file")

    data_dir = sys.argv[1]
    output = open(sys.argv[2], 'a')
    files = open(sys.argv[2], 'r').readlines()
    files = [f.split(';')[0].replace('\n', '') for f in files]

    print(files)

    # walk through a directory
    for dirpath, dirnames, filenames in os.walk(data_dir):

        for file in filenames:
            file = file.replace("._", "")
            fp = os.path.join(dirpath, file)

            if fp in files:
                print("skipped: ", fp)

            elif file.endswith('tif') and ("GFP" in file or "CY5" in file):

                try:
                    # show image
                    img = cv2.imread(fp)
                    cv2.imshow("image", img)

                    # waits for user to press any key
                    # (this is necessary to avoid Python kernel form crashing)
                    cv2.waitKey(0)

                    # closing all open windows
                    cv2.destroyAllWindows()

                except Exception as e:
                    print("Error: ", e)
                    print("File: ", fp)

                    ans = input("Continue ? (y/n) : ")
                    while ans != "y":
                        ans = input("Continue ? (y/n) : ")
                    continue

                # ask for classification
                classification = input(f"{file}: infected? (0 / 1): ")

                # check that input is valid
                while classification not in ["0", "1"]:
                    classification = input(f"{file}: infected? (0 / 1): ")

                # add classification data to txt file
                line = f"{fp};{classification}\n"
                output.write(line)


if __name__ == "__main__":
    main()

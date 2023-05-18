import os
import sys
import subprocess
from datetime import datetime

import cv2
import csv
from tqdm import tqdm


def load_data_from_df(dataframe, img_size=256):
    """
    load data from a class file; returns a tuple of (images, labels)
    """
    print("loading images and labels...")

    images = []
    labels = []

    seen = set()

    for path, label in tqdm(dataframe, total=len(dataframe)):
        label = int(label.strip())

        # ignore files that don't exist, are in the skip list, who's directory is in the skip list
        # or have ._ in their name
        if os.path.isfile(path) and path not in seen and '._' not in path:
            # parse label
            labels.append(label)

            # parse image as ndarray
            im = cv2.imread(path)

            if im is not None:
                # resize image
                resizeIM = cv2.resize(im, (img_size, img_size))
                images.append(resizeIM)

            seen.add(path)

    return (images, labels)


class FileHandling:
    """ File handling class for the GUI"""

    def __init__(self):
        self.dir = None
        self.saveFile = None
        self.logDir = None
        self.imgExtensions = [
            ".png", ".tif", ".jpg"
        ]
        self.skips = ["._"]

    def getImagePaths(self):
        """
        extract all image files from a directory
        """
        img_paths = []

        for path, dirnames, filenames in os.walk(self.dir):
            for filename in filenames:
                name, ext = os.path.splitext(filename)
                if ext in self.imgExtensions and not any(skip in name for skip in self.skips):
                    img_paths.append(os.path.join(path, name+ext))

        return img_paths

    def saveResults(self, results):
        logfile = open(self.saveFile, "w")
        logfile.write(f"images;labels\n")
        for key in results.keys():
            logfile.write(f"{key};{results[key]}\n")
        logfile.close()

    def isImageFile(self, path):
        return os.path.isfile(path) and any([ext in path for ext in self.imgExtensions])

    def isModelFile(self, path):
        ext = os.path.splitext(path)[1]
        return ext == ".h5"

    def isTxtFile(self, path):
        txt_exts = [".csv", ".txt"]
        ext = os.path.splitext(path)[1]
        return os.path.isfile(path) and ext in txt_exts

    def openImage(self, path):
        imageViewerFromCommandLine = {'linux': 'xdg-open',
                                      'win32': 'explorer',
                                      'darwin': 'open'}[sys.platform]
        subprocess.run([imageViewerFromCommandLine, path])

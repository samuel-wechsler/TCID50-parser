import os
import sys
import subprocess
from datetime import datetime


class FileHandling:
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
        logfile.write(f"files;labels\n")
        for key in results.keys():
            logfile.write(f"{key};{int(results[key])}\n")
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

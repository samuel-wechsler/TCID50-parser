# TCID50-parser
Overview:
This GitHub repository contains the code for a project that aims to automate the process of evaluating fluorescence signals in cell cultures to calculate the TCID50 of an endpoint dilution assay. This project is intended to help researchers and scientists to save time and effort in manually performing these calculations, and to improve the accuracy and consistency of the results.

Getting Started:
To use this project, you will need to have Python 3.6 or higher installed on your computer. Additionally, you will need to install the required Python packages, which are listed in the "requirements.txt" file.

To run the code, navigate to the "src" directory and run the desired Python file. The available modules and their purposes are:

classify.py: This module was used to classify data by hand, specifically data which the model was trained on.
data_eng.py: This module parses microscopy images so that they end up having the correct filenames, filetypes, and locations.
test_train.py: This module was used to create and train a convolutional neural network.
evaluate.py: This module contains functions that can evaluate a fluorescence signal of a cell culture image and calculate the TCID50 of a well plate (represented by a matrix).
Code Structure:
The code for this project is organized into four modules, which are located in the "src" directory. These modules include:

classify.py: This module was used to classify data by hand, specifically data which the model was trained on. This module uses functions to load and preprocess data from a specified directory and creates CSV files for use in training and testing the model.
data_eng.py: This module parses microscopy images so that they end up having the correct filenames, filetypes, and locations. This is achieved using functions that read and rename files, and move them to the specified directory.
test_train.py: This module was used to create and train a convolutional neural network. This module uses functions to preprocess images, split the data into training and testing sets, train the model, and save the resulting model.
evaluate.py: This module contains functions that can evaluate a fluorescence signal of a cell culture image and calculate the TCID50 of a well plate (represented by a matrix). This module uses functions to load images, preprocess them, and calculate the TCID50 values for the well plate.
Contributing:
We welcome contributions to this project! If you notice any bugs or have suggestions for new features, please create an issue on the GitHub repository. If you would like to contribute code, please fork the repository and submit a pull request with your changes.

License:
This project is licensed under the MIT License. Please see the LICENSE file for more details.

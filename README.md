# TCID50-parser

## Overview:
This GitHub repository contains the code for a project that aims to automate the process of evaluating fluorescence signals in cell cultures to calculate the TCID50 of an endpoint dilution assay.

## Getting Started:
To use this project, you will need to have Python 3.8 or higher installed on your computer. The github repository can be downloaded using git:
````
git clone https://github.com/samuel-wechsler/TCID50-parser.git
````

Then navigate to the "src" directory and run the desired Python file. The main modules and their command line instructions are:
### evaluate.py
In order to predict if a single image shows an infected or non-infected cell culture, run:
````
python evaluate.py -f evaluate_image -p path/to/image -m path/to/model/dir
````

To determine the TCID_{50} of an endpoint dilution assay, enter the following command line argument:
````
python evaluate.py -f evaluate_dir -d path/to/image/dir -m path/to/model/dir
````

### train_test.py
Training a new convolutional neural network model requires splitting all image data into two subdirectories called "infected" and "not_infected" (e.g., using classify.py). Then run the following command line argument to fit the model:
````
python train_test.py -f train -p path/to/image/dir -m [path/to/model/save/dir]
````

## Code Structure:
The code for this project is organized into four modules, which are located in the "src" directory. These modules include:

- classify.py: This module was used to classify data by hand, specifically data which the model was trained on. This module uses functions to load and preprocess data from a specified directory and creates CSV files for use in training and testing the model.

- data_pipe.py: This module parses microscopy images so that they end up having the correct filenames, filetypes, and locations. This is achieved using functions that read and rename files, and move them to the specified directory.

- train_test.py: This module was used to create and train a convolutional neural network. This module uses functions to preprocess images, split the data into training and testing sets, train the model, and save the resulting model.

- evaluate.py: This module contains functions that can evaluate a fluorescence signal of a cell culture image and calculate the TCID50 of a well plate (represented by a matrix). This module uses functions to load images, preprocess them, and calculate the TCID50 values for the well plate.

## Contributing:
Contributions to this project are always welcome! If you notice any bugs or have suggestions for new features, please create an issue on the GitHub repository. If you would like to contribute code, please fork the repository and submit a pull request with your changes.

## Future Ideas
- improve data normalization -> find automatic way of adjusting brightness / contrast level given some negative control such that fluorescent noise gets filtered.
- implement K-fold cross-validation


## License:
This project is licensed under the MIT License. Please see the LICENSE file for more details.

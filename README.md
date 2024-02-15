# TCID50-parser
## A cell image analysis tool for automated titration readout of endpoint dilution assays

This program was developed to improve the workflow for titration readout of endpoint dilution assays. It provides the following functionalities:
* Automated classification of fluorescence microscopy images of cell cultures using a convolutional neural network
* Manual classification of cell culture images
* Calculation of the Tissue-Culture-Infectious-Dose-50% (TCID50) using the Spearman-Kärber method
* Development of custom convolutional neural networks for automated classification of cell culture images, perhaps using transfer learning

## See it in action
Here's the basics of how to use the program.
<p align="left">
  <img src="https://github.com/samuel-wechsler/TCID50-parser/assets/98318988/a9986014-edd8-4bb3-96d5-24951948801a" width="500"/>
</p>

Here's a visual illustration of what the program's output.
<p align="left">
  <img width="500" src="https://github.com/samuel-wechsler/TCID50-parser/assets/98318988/71f8ed4f-91f9-4fcc-bf0c-c10efbc9ec9e">
</p>

## Getting started
To install this programm, you will need to have Python 3.8 or higher installed. Download the programm using git:
````
git clone https://github.com/samuel-wechsler/TCID50-parser.git
````

Pick up dependencies using pip:
````
pip install -r requirements.txt
````

After the successfull installation, nagivate to the source directory and run:
`````
python TCID50_parser.py
`````

## User Guide

### Classifying images manually
To classify images manually, drag and drop the desired images onto the display area. Then use the buttons to classify the images as positive or negative, or use left- and right-arrow keys as shortcuts. Press ctrl+z to undo the last classification.

### Classifying images automatically
Click onto the "Classify" button to classify all images in the current directory. Enter specifications of plates and wells when ask by the program.


### Training models
#### Proof of Concept
Due to the failure of the convolutional neural network to generalize to new data, the program supports transfer learning to improve the perfomance of the model.

Perfomance before transfer learning:
`````
accuracy: 0.875
true positive: 1.0
true_negative: 0.8261
`````

Perfomance after transfer learning on 100 images:
`````
accuracy: 0.9792
true positive: 0.9630
true_negative: 0.9855
`````

#### Creating a new model
Switch to the "Train" tab to train a new model. Drag and drop a classifications text file onto the display area. The text file should be formatted as follows:
````
files;labels
image1.png;1
image2.png;0
`````
Then specify training parameters, click onto the "Train" button. The program will train a convolutional neural network using the specified parameters and save the model to whichever directory you specify.

## Known issues (Work in progress)
* The program doesn't yet support uploading image data or classifications thereof as well as model architectures.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details

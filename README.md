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
  <img src="https://github.com/samuel-wechsler/TCID50-parser/assets/98318988/985b8c4f-b264-4a3a-bea1-694a6860796c)" width="500"/>
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

### Proof of Concept
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



## How to contribute
If you found an issue or would like to submit an improvement, please submit an issue using the issue tab. If you would like to submit a pull request with a fix or improvement, please reference the issue you created!

## Known issues (Work in progress)
* The program doesn't yet support uploading image data or classifications thereof as well as model architectures.

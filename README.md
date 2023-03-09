# TCID50-parser
This repository is a project with the aim of automating the process of evaluating TCID50 assays.

There are three categories of programms in this repository:
1. Code that parses raw data obtained by microscopy and creates copies of them with the appropriate filenames in a second directory.
2. A convolutional neural network that classifies whether or not a given cell culture is infected based on fluorescence signals.
3. Code that evaluates a well plate (indicating whether or not wells are infected) and then calculates the TCID50 using the Spearman-Kärber-Method.


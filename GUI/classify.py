import os
import numpy as np

from scipy.stats import binom

from PIL import Image, ImageDraw, ImageFont

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa
import tensorflow as tf
import tensorflow_io as tfio

IMG_SHAPE = 256


class ClassifyUtils:
    """ utils for image classification """

    def __init__(self, model):
        self.model = model

    def classify_image(self, image_path, classnames=[0, 1]):
        """ Classify an image with a specified model """
        prediction = self.model.predict(
            self.load_and_prep_image(image_path), verbose=0
        )

        if len(prediction[0]) > 1:
            # return most likely class of prediction
            return classnames[int(tf.round(prediction)[0][0])], max(prediction[0])

        else:
            ### PROBABILY STILL BUGGY ###
            print(prediction[0][0])
            return classnames[int(prediction[0][0] >= 0.5)], 2*(max(prediction[0][0], 1 - prediction[0][0]) - 0.5)

    def load_and_prep_image(self, image_path):
        """
        reading and resizing of image into an appropriate tensor
        """
        img = tf.io.read_file(image_path)

        # Decode the read file into a tensor & ensure 3 colour channels
        if image_path.endswith("tif"):
            img = tfio.experimental.image.decode_tiff(img)
            img = tfio.experimental.color.rgba_to_rgb(img)
        else:
            img = tf.image.decode_image(img, channels=3)

        # Resize the image
        img = tf.image.resize(img, size=[IMG_SHAPE, IMG_SHAPE])
        img = tf.expand_dims(img, axis=0)

        return img


class Classify:
    """ """


class ClassifyPlates:
    """ Classifies all plates in a specified directory """

    def __init__(self, params):
        self.params = params
        self.params.model = tf.keras.models.load_model(params.model_path)

        self.plate_dirs = os.listdir(self.params.plates_dir)
        self.plates = self.get_plates()

    def get_plates(self):
        """
        Classifies all plates in the specified directory
        """
        plates = []
        for plate_dir in self.plate_dirs:
            # get full path of plate
            plate_dir = os.path.join(self.params.plates_dir, plate_dir)

            # ignore files in directory (plates must be subdirs of dir)
            if os.path.isdir(plate_dir):
                plate = Plate(params=self.params, dir=plate_dir)
                plates.append(plate)

        return plates

    def classify_plates(self, thread):
        c = 1
        for plate in self.plates:
            plate.classify()
            plate.display()
            # TODO: refine progress updates, currently only updates after plate is classified
            # emit progress as percentage
            thread.classify_complete.emit(round(c / len(self.plates) * 100))
            c += 1

    def get_classifications(self):
        """
        returns a dictionary with key of each file_path and value of labels
        """
        classifications = dict()

        for plate in self.plates:
            for row in range(len(plate.classified_plate)):
                for col in range(len(plate.classified_plate[row])):
                    classifications[plate.image_paths[row][col]
                                    ] = plate.classified_plate[row][col]

        # sort dictionary by value (true or false)
        classifications = dict(
            sorted(classifications.items(), key=lambda item: not item[1])
        )

        return classifications

    def add_manual_checks(self, manual_checks):
        for plate_dir in manual_checks.keys():
            plate = [plate for plate in self.plates if plate.dir == plate_dir][0]
            row, col, label = manual_checks[plate_dir]
            plate.classified_plate[row][col] = label


class Plate(ClassifyUtils):
    def __init__(self, params, dir):
        super().__init__(params.model)

        self.params = params

        # define rows and columns
        self.rows = [
            chr(i) for i in range(ord(params.row_range[0]), ord(params.row_range[1]) + 1)
        ]
        self.cols = list(range(params.col_range[0], params.col_range[1] + 1))

        self.classified_plate = np.zeros((len(self.rows), len(self.cols)))

        self.dir = dir
        self.image_paths = self.get_image_paths()

        # parameters of endpoint dilution assay
        self.titer = None

        # evaluated parameters
        self.classified_plate = None
        self.low_confidence = None
        self.outliers = None

    def get_image_paths(self):
        """
        returns a matrix of found images in the specified path
        """
        ### TODO: HOW TO DEAL WITH MISSING IMAGES? ###
        images = [
            img for img in os.listdir(self.dir) if any(img in path for path in self.params.img_paths)
        ]

        plate = []

        for row_nb in self.rows:
            row = []
            for col_nb in self.cols:
                coord = f"{row_nb}{col_nb}"

                # find image with specified coordinates
                matches = [image for image in images if coord in image]
                file_path = os.path.join(self.dir, matches[0])
                row.append(file_path)

            plate.append(row)

        return plate

    def classify(self):
        """
        This method takes a matrix of paths to images as an argument, classifys each image with a specified model
        and returns the predictions.
        """
        classified = self.params.classified

        self.classified_plate = []
        self.low_confidence = []

        for row_idx in range(len(self.image_paths)):
            classified_row = []
            for col_idx in range(len(self.image_paths[row_idx])):
                if self.image_paths[row_idx][col_idx] in classified.keys():
                    # if image has been manually classified, append to row
                    classified_row.append(
                        classified[self.image_paths[row_idx][col_idx]]
                    )
                    print("skiped", self.image_paths[row_idx][col_idx])
                    continue

                # classify image with model
                prediction = self.classify_image(
                    self.image_paths[row_idx][col_idx]
                )

                result, confidence = prediction

                # print(os.path.basename(
                #     self.image_paths[row_idx][col_idx]), result, confidence)

                # append result to row
                classified_row.append(result)

                # if low confidence, append to low_confidence list
                if confidence < 0.55:
                    self.low_confidence.append((row_idx, col_idx))

            self.classified_plate.append(classified_row)

    def get_titer(self):
        """
        Calculates TCID50 according to spearman and karber formula
        """
        d_0 = self.initial_dilution
        d = self.serial_dilution

        # special case: no fully infected rows
        if not any([sum(row) == len(row) for row in self.classified_plate]):
            d_0 += 0.5

        # find fully infected row with greatest dilution
        row0 = self.most_diluted()

        # calculate the log10 concentration of the first fully infected row
        x_0 = d_0 - (row0) * d

        # calculate sum of fractions of infected wells per row
        s = 0

        # smooth out data
        plate = sorted(plate, key=lambda row: (
            sum(row) / len(row)), reverse=True)

        # remove duplicates
        p = []
        [p.append(row) for row in plate if row not in p]

        for row in p:
            s += (sum(row) / len(row))

        self.titer = 10 ** -((x_0 + d/2 - d * s) + d_0)

    def most_diluted(self):
        """ helper of get_titer """
        row0 = 0
        for row in range(len(self.classified_plate)):
            if sum(self.classified_plate[row]) == len(self.classified_plate[row]):
                row0 = row
        return row0

    def get_outliers(self):
        """
        returns all rows that are considered to be outliers (i.e., p<0.05).
        """
        self.outlier_rows = []
        d = 1

        # Overestimate virus count
        virus_count = 1.5 * self.get_titer()

        for row in self.classified_plate:
            # get probability of CPE
            p = self.prob_cpe(virus_count, d)

            # get probability of current row distribution
            r = binom.pmf(k=sum(row), n=len(row), p=p)

            # if outlier, append row index
            if r < 0.05:
                self.outlier_rows.append(d-1)
            d += 1

    def prob_cpe(self, titer, dilution):
        """
        calculates the probablity of CPE occuring given specified parameters
        """
        return 1 - np.exp(-titer / (self.particels_to_pfu * np.power(self.serial_dilution, dilution)))

    def manual_corrections(self):
        """
        returns plate directory, row and column of all wells that need to be rechecked manually
        """
        manual_corrections = self.low_confidence.copy()
        manual_corrections.extend(
            (self.dir, (outlier, i)) for outlier in self.outliers for i in range(len(self.cols))
        )
        return manual_corrections

    def save(self, save_path):
        """
        saves the plate as a visual representation
        """
        well_size = 100
        well_border = 10

        nb_col = len(self.plate[0])
        nb_row = len(self.plate)

        # Create a blank canvas
        img = Image.new("RGBA",
                        ((nb_col + 5) * well_size,
                         (nb_row + 1) * well_size)
                        )

        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(1, nb_row + 1):
            # mark rows with an asterisk if they are considered to be outliers
            outlier = "    *" if (i-1) in self.outliers else ""

            # draw row letter
            draw.text((0, i * well_size), outlier, fill="black", font=font)
            draw.text((0, i * well_size),
                      self.rows[i-1], fill="black", font=font)

            for j in range(1, nb_col + 1):
                # draw column number
                draw.text((j * well_size + 2 * well_border, 0),
                          str(self.cols[j-1]), fill="black", font=font)

                # define dimensions of well
                dims = ((j * well_size + well_border, i * well_size + well_border),
                        ((j + 1) * well_size - well_border, (i + 1) * well_size - well_border))

                # draw well
                draw.rectangle(dims, fill="white", outline="black", width=3)

                # draw well content
                if self.classified_plate[i-1][j-1] == 1:
                    cross = (dims,
                             ((j * well_size + well_border, (i + 1) * well_size - well_border),
                              ((j + 1) * well_size - well_border, i * well_size + well_border))
                             )

                    draw.line(cross[0], fill="black", width=3)
                    draw.line(cross[1], fill="black", width=3)

        # draw titer
        draw.text((((nb_row) * well_size / 2),
                   (nb_col + 3) * well_size),
                  f"TCID50={self.titer}",
                  fill="black",
                  font=font
                  )

        img.save(save_path)

    def display(self):
        """
        prints a visual representation of the plate
        """
        n = len(self.classified_plate[0])
        print("\n\n" + n * "----")
        for row in self.classified_plate:
            print('|', end="")
            for well in row:
                val = " X" if well == 1 else "  "
                print(val, end="")
                print(" |", end="")
            print("\n" + n * "----")
        print("\n\n")

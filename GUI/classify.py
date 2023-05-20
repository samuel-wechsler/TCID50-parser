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
            return classnames[int(prediction[0][0] >= 0.5)], 1 - abs(prediction[0][0] - 0.5) * 2

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
            # also ingore plate dirs that were filtered out
            if os.path.isdir(plate_dir) and any(plate_dir in path for path in self.params.img_paths):
                plate = Plate(params=self.params, dir=plate_dir)
                plates.append(plate)

        return plates

    def classify_plates(self, thread):
        """ calls classify method of each plate """
        # calculate total number of images to classify
        length1 = ord(self.params.row_range[1]) - \
            ord(self.params.row_range[0]) + 1
        length2 = self.params.col_range[1] - self.params.col_range[0] + 1
        total = len(self.plates) * length1 * length2

        counter = 1

        for plate in self.plates:
            plate.classify(
                update_progress=thread.classify_complete, counter=counter, total=total)

            counter += length1 * length2

    def get_unclassified_images(self):
        """
        returns a dictionary of all images, sorted by classification (first all unclassified images, then all classified images)
        """
        classifications = dict()

        for plate in self.plates:
            for row in range(len(plate.classified_plate)):
                for col in range(len(plate.classified_plate[row])):
                    classifications[plate.image_paths[row][col]
                                    ] = plate.classified_plate[row][col]

        # sort dictionary by None values
        classifications = dict(
            sorted(classifications.items(),
                   key=lambda item: (item[1] is not None, item[1]))
        )

        return classifications

    def get_manual_checks(self):
        """
        returns a dictionary of all images that need to be rechecked manually
        """
        manual_checks = dict()

        for plate in self.plates:
            for image, (row, col) in plate.manual_recheck_list:
                manual_checks[image] = (row, col)

        return manual_checks

    def set_manual_checks(self, manual_checks):
        """ This method adds images were rechecked manually to the classified dictionary """
        for image in manual_checks.keys():
            plate_dir = os.path.dirname(image)

            plate = [plate for plate in self.plates if plate.dir == plate_dir][0]

            row, col, label = manual_checks[image]
            plate.classified_plate[row][col] = label

            print(row, col, plate.dir)

    def get_titers(self):
        """ calculates the titer of each plate """
        titers = dict()

        for plate in self.plates:
            titers[plate.get_name()] = plate.get_titer()

        return titers

    def save_plates(self, save_dir):
        """ saves all plates in a specified directory """
        for plate in self.plates:
            plate.save(os.path.join(save_dir, plate.get_name() + ".png"))


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
        self.low_confidence = []
        self.outliers = []
        self.manual_recheck_list = []

    def get_name(self):
        """ returns the name of the plate: dir_name_row_range_col_range """
        return f"{os.path.basename(self.dir)}_{self.rows[0]}{self.rows[-1]}{self.cols[0]}{self.cols[-1]}"

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
                if len(matches) == 0:
                    print(f"missing image: {plate.dir} {coord}")
                file_path = os.path.join(self.dir, matches[0])
                row.append(file_path)

            plate.append(row)

        return plate

    def classify(self, update_progress, counter, total):
        """
        This method takes a matrix of paths to images as an argument, classifys each image with a specified model
        and returns the predictions.
        """
        classified = self.params.classified

        self.classified_plate = []

        for row_idx in range(len(self.image_paths)):
            classified_row = []
            for col_idx in range(len(self.image_paths[row_idx])):
                # update progress
                update_progress.emit(counter / total * 100)
                counter += 1

                # if image has been manually classified, add manual classification to row
                if self.image_paths[row_idx][col_idx] in classified.keys():
                    classified_row.append(
                        classified[self.image_paths[row_idx][col_idx]]
                    )
                    print("skiped", self.image_paths[row_idx][col_idx])
                    continue

                # classify image with model
                result, confidence = self.classify_image(
                    self.image_paths[row_idx][col_idx]
                )

                # append result to row
                classified_row.append(result)

                # if low confidence, set classification to None*
                if confidence > 0.8:
                    self.low_confidence.append(
                        (self.image_paths[row_idx]
                         [col_idx], (row_idx, col_idx))
                    )
            self.classified_plate.append(classified_row)

        # TODO: code smell - refactor
        self.get_manual_recheck_list()

    def get_titer(self):
        """
        Calculates TCID50 according to spearman and karber formula

        Note: weird variable names
        d_0: negative log10 of initial dilution
        d: log10 of serial dilution
        x_0: log10 concentration of first fully infected row
        """
        d_0 = -np.log10(self.params.initial_dilution)
        d = np.log10(self.params.serial_dilution)

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
        self.classified_plate = sorted(self.classified_plate, key=lambda row: (
            sum(row) / len(row)), reverse=True)

        # remove duplicates
        p = []
        [p.append(row) for row in self.classified_plate if row not in p]

        for row in p:
            s += (sum(row) / len(row))

        self.titer = 10 ** -((x_0 + d/2 - d * s) + d_0)
        return self.titer

    def most_diluted(self):
        """ helper of get_titer """
        row0 = 0
        for row in range(len(self.classified_plate)):
            if sum(self.classified_plate[row]) == len(self.classified_plate[row]):
                row0 = row
        return row0

    def get_outlier_rows(self):
        """
        returns all rows that are considered to be outliers (i.e., p<0.05).
        """
        outlier_rows = []
        d = 1

        virus_count = self.get_titer()

        for row in self.classified_plate:
            # get probability of CPE
            p = self.prob_cpe(virus_count, d)

            # get probability of current row distribution
            r = binom.pmf(k=sum(row), n=len(row), p=p)

            # if outlier, append row index
            if r < 0.05:
                outlier_rows.append(d-1)
            d += 1

        return outlier_rows

    def prob_cpe(self, titer, dilution):
        """
        calculates the probablity of CPE occuring given specified parameters
        """
        return 1 - np.exp(-titer / (self.params.particle_to_pfu * np.power(self.params.serial_dilution, dilution)))

    def get_manual_recheck_list(self):
        """
        returns plate directory, row and column of all wells that need to be rechecked manually;
        i.e., all wells that were classified with low confidence and all wells that are considered to be outliers
        """
        # get outliers
        outliers = self.get_outlier_rows()

        # get manual corrections
        self.manual_recheck_list = self.low_confidence.copy()
        self.manual_recheck_list.extend(
            (self.image_paths[outlier][i], (outlier, i)) for outlier in outliers for i in range(len(self.cols))
        )

        # set outlier / low confidence wells to None
        for (_, (row, col)) in self.manual_recheck_list:
            self.classified_plate[row][col] = None

        return self.manual_recheck_list

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

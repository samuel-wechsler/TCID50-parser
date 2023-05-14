import csv
import os
from PIL import Image
import cv2
import shutil


def adapt_filenames(input_dir):
    """
    This function was used to change filenames of Laloli dataset images s.t.
    it reflects the infection state.
    """
    for dirpath, dirnames, filenames in os.walk(input_dir):
        for filename in filenames:
            old = os.path.join(dirpath, filename)
            if 'M' in filename:
                print(filename[:-4] + '_ni.png')
                new = os.path.join(dirpath, filename[:-4] + '_ni.png')
            else:
                print(filename[:-4] + '_i.png')
                new = os.path.join(dirpath, filename[:-4] + '_i.png')

            os.rename(old, new)


def adapt_filenames_2(input_dir, classifications):
    """
    This function was used to change filenames of matura dataset images s.t.
    it reflects the infection state.
    """
    with open(classifications, 'r') as f:
        for line in f:
            # separating parsed line
            data = line.split(';')

            # parse filename
            filename = data[0]
            print(filename)
            label = int(data[-1])

            suffix = '_i.png' if label == 1 else '_ni.png'

            old = os.path.join(input_dir, filename)
            new = os.path.join(input_dir, filename[:-4] + suffix)

            try:
                os.rename(old, new)
            except:
                continue


def adapt_filenames_3(input_dir, filetype="png"):
    filenames = os.listdir("datasets/Laloli_et_all2022_raw_images/infected")
    filenames.extend(os.listdir(
        "datasets/Laloli_et_all2022_raw_images/not_infected"))

    for filename in filenames:

        # check if file is GFP image
        if filename.endswith('.tif') and "GFP" in filename:
            new_filename = filename[:-4]+'.png'

            if "M" in filename:
                dst = os.path.join(input_dir, "not_infected", filename)
                src = os.path.join(input_dir, "not_infected", new_filename)
            else:
                dst = os.path.join(input_dir, "infected", filename)
                src = os.path.join(input_dir, "infected", new_filename)

            im = Image.open(dst)
            im.save(src)


def adapt_filenames_4():
    input_dir = "datasets/Laloli_et_all2022_raw_images/not_infected"
    session_dirs = "datasets/Laloli_et_all2022_raw_images"

    dir_paths = os.walk(session_dirs)

    dir_paths = [dir[0] for dir in dir_paths]

    # for dir in dir_paths:
    #     print(dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.tif') and "GFP" in filename:
            meta = filename.split('_')

            # print("_".join(meta[3:5]))
            # print("_".join(meta[:3]))

            found_dir = [dir for dir in dir_paths if "_".join(
                meta[:3]) in dir and "_" + "_".join(meta[3:5]) == dir.split("/")[-1]]

            print(filename)
            assert len(found_dir) == 1

            src = os.path.join(input_dir, filename)
            dst = os.path.join(found_dir[0], filename)

            print(src)
            print(dst)
            shutil.move(src, dst)


def parse_files_old(input_dir, output_dir, filetype="png"):
    """
    This function was used to parse all images from the matura dataset into
    distinct categories (merge, gfp, phase contrast).
    """
    for filename in os.listdir(input_dir):
        if filename in [".DS_Store"]:
            continue
        src = os.path.join(input_dir, filename)

        if "merge" in filename:
            dst = os.path.join(output_dir, "merge",
                               filename.split('.')[0] + '.' + filetype)

        elif "GFP" in filename:
            dst = os.path.join(output_dir, "GFP", filename.split('.')[
                               0] + '.' + filetype)

        elif "PhaseContrast" in filename:
            dst = os.path.join(output_dir, "PhaseContrast",
                               filename.split('.')[0] + '.' + filetype)

        im = Image.open(src)
        im.save(dst)


def load_data_old(data_dir, classficiations):
    """
    This function loads data from a data directory and their corresponding classficications
    from a given text file.
    The function the returns both lists in a tuple.
    """
    images = []
    labels = []

    with open(os.path.join(classficiations), 'r') as f:
        for line in f:

            # parse filename and path
            data = line.split(';')
            filename = data[0]
            path = os.path.join(data_dir, filename)

            if os.path.isfile(path):
                print("parsing line ", line)

                # parse label
                label = data[-1]
                labels.append(label)

                # parse image as ndarray
                im = cv2.imread(path)

                # resize image
                resizeIM = cv2.resize(im, (256, 256))
                print(resizeIM.shape)
                images.append(resizeIM)

    return (images, labels)


def remove_duplicates(class_file, new_class_file):
    file = open(new_class_file, 'w')
    added = []

    with open(class_file, 'r') as f:
        lines = f.readlines()

        for line in lines:
            if line not in added:
                file.write(line)
                added.append(line)


def add_classifications(add_dir, class_file):

    f = open(class_file, 'w')

    # walk through all files in a directory
    for dirpath, dirnames, filenames in os.walk(add_dir):

        for filename in filenames:

            if filename.endswith('png'):
                # get path of file
                path = os.path.join(dirpath, filename)

                if 'not_infected' in dirpath:
                    entry = f"{path};{0}\n"
                    f.write(entry)
                else:
                    entry = f"{path};{1}\n"
                    f.write(entry)


# add_classifications("/Volumes/T7/tcid50_datasets/matura_data/GFP",
#                     'classficiations/matura_classifications.txt')
# add_classifications("/Volumes/T7/tcid50_datasets/Laloli_et_all2022/Laloli_et_all2022_raw_png",
#                     'classficiations/laloli_classifications.txt')


def classify_phase_contrast():
    classes = open("classifications/probst_classification.txt", "r")
    classes = {
        i.split(";")[0][:-19]: int(i.split(";")[1].replace("\n", "")) for i in classes
    }

    dir = "/Volumes/T7/tcid50_datasets/Lukas_Probst_IDV/Lukas_Probst_IDV_ori"

    c = 0
    # walk through all files in a directory
    for dirpath, dirnames, filenames in os.walk(dir):

        for filename in filenames:

            if "Bright Field" in filename and '._' not in filename:
                path = os.path.join(dirpath, filename)

                if (path[:-28] in classes.keys()):
                    print(f"{path};{classes[path[:-28]]}")
                    c += 1
                    print(c)


# def train_model_v2(train_dir):
#     print("training model")
#     # Create a data generator
#     df = pd.read_csv(train_dir, delimiter=";", names=["filenames", "labels"])
#     df["labels"] = df["labels"].astype("str")

#     print(df.columns)
#     print(df.dtypes)

#     train_datagen = ImageDataGenerator(rescale=1./255,
#                                        rotation_range=20,
#                                        width_shift_range=0.2,
#                                        height_shift_range=0.2,
#                                        shear_range=0.2,
#                                        zoom_range=0.2,
#                                        horizontal_flip=True,
#                                        fill_mode='nearest',
#                                        validation_split=TEST_SIZE
#                                        )

#     # Load the data
#     training_data = train_datagen.flow_from_dataframe(dataframe=df,
#                                                       directory=None,
#                                                       x_col="filenames",
#                                                       y_col="labels",
#                                                       subset="training",
#                                                       batch_size=95,
#                                                       shuffle=True,
#                                                       target_size=(IMG_HEIGHT,
#                                                                    IMG_WIDTH),
#                                                       class_mode='binary')

#     validation_data = train_datagen.flow_from_dataframe(dataframe=df,
#                                                         directory=None,
#                                                         x_col="filenames",
#                                                         y_col="labels",
#                                                         subset="validation",
#                                                         batch_size=95,
#                                                         shuffle=True,
#                                                         target_size=(IMG_HEIGHT,
#                                                                      IMG_WIDTH),
#                                                         class_mode='binary')

#     step_size_train = training_data.n
#     step_size_valid = validation_data.n

#     print(step_size_train, step_size_valid)

#     model = get_model()
#     model.fit(training_data,
#               steps_per_epoch=step_size_train,
#               validation_data=validation_data,
#               validation_steps=step_size_valid,
#               epochs=EPOCHS,
#               verbose=2)

# import numpy as np
# a = [3.16E7, 1.47E6, 4.64E3, 6.81E3, 1E6, 1.47E7, 2.15E6, 1E6, 4.64E4, 4.64E4]
# b = [2.15E7, 1E6, 3.16E3, 4.64E3, 4.64E5, 1E7, 1.47E6, 4.64E5, 4.64E4, 3.16E4]

# c = [np.log10(x) - np.log10(y) for x, y in zip(a, b)]

# c.extend([0]*(38*2-10))

# print(c, len(c))

# print(np.mean(c))
# print(np.std(c))

# sort csv file
f = open("classifications/classification.txt")
seen = []

for l in f:
    l = l.split(";")
    path = l[0].split('/')
    l = path[-2] + '/' + path[-1]

    if l in seen:
        print(l)

    seen.append(l)

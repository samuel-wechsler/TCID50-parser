import os
import sys
from skimage.io import imread, imsave
from skimage import exposure

from PIL import Image

def main():
    if len(sys.argv) not in [4, 5]:
        sys.exit("Usage: python data_eng.py input/dir output/dir function")

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if sys.argv[3] == "parse":
        parse_files(input_dir, output_dir)
            

def parse_files_old(input_dir, output_dir, filetype="png"):

    for filename in os.listdir(input_dir):
        if filename in [".DS_Store"]:
            continue
        src = os.path.join(input_dir, filename)

        if "merge" in filename:
            dst = os.path.join(output_dir, "merge", filename.split('.')[0] + '.' + filetype)
        
        elif "GFP" in filename:
            dst = os.path.join(output_dir, "GFP", filename.split('.')[0] + '.' + filetype)

        elif "PhaseContrast" in filename:
            dst = os.path.join(output_dir, "PhaseContrast", filename.split('.')[0] + '.' + filetype)
        
        im = Image.open(src)
        im.save(dst)

def parse_files(input_dir, output_dir, filetype="png"):

    for dirpath, dirnames, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('.tif') and "GFP" in filename:
                src = os.path.join(dirpath, filename)

                paths = dirpath.split('/')
                infos = [path.split('_') for path in paths]
                meta = []
                meta.extend(infos[2][-3:])
                meta.extend(infos[3][-2:])

                filename = "_".join(meta) + "_GFP"


                dst = os.path.join(output_dir, filename + '.' + filetype)


                print(f"converting {filename}")

                image = imread(src)
                image_dark = exposure.adjust_gamma(image, gamma=2,gain=1)
                imsave(dst, image_dark)
                
                

if __name__ == "__main__":
    main()

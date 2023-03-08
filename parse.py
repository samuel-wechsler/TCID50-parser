import os
import sys
import shutil

# parse data
parsed_dir = "/Users/samuelwechsler/Documents/Gymer/MA/MA Schriftlich/RGBs_copy"
dst_dir = "/Users/samuelwechsler/Documents/TCID50-parser/data"

def main():

    if len(sys.argv) != 3:
        sys.exit("Usage: python parse.py input/dir output/dir")

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    for filename in os.listdir(input_dir):
        src = os.path.join(input_dir, filename)

        if "merge" in filename:
            dst = os.path.join(output_dir, "merge", filename)
            shutil.copyfile(src, dst)
        
        elif "GFP" in filename:
            dst = os.path.join(output_dir, "GFP", filename)
            shutil.copyfile(src, dst)

        elif "PhaseContrast" in filename:
            dst = os.path.join(output_dir, "PhaseContrast", filename)
            shutil.copyfile(src, dst)


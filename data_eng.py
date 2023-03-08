import os
import sys

def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python parse.py input/dir output/dir")

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    output = open(output_dir, "w")

    with open(input_dir, "r") as f:
        for line in f:
            data = line.split(";")

            filename = data[0]
            filename = filename[:-11] + '_GFP.tif'

            if os.path.isfile(os.path.join("data", "GFP", filename)):
                new_data = [filename]
                new_data.extend(data[1:])
                output.write(";".join(new_data))
            

            # gfp = ["gfp"]
            # gfp.extend(data)

            # merge = ["merge"]
            # merge.extend(data)

            # output.write(";".join(gfp))
            # output.write(";".join(merge))

if __name__ == "__main__":
    main()
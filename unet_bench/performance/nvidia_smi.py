from datetime import datetime
from mimetypes import init
import os
import numpy as np

def main():
    """
    A version of hl-smi.py, this file cleans all of the hl-smi generated from Theta at once, located in theta-csvs.
    """

    # Allow script to work when called from anywhere.
    location = os.path.dirname(os.path.abspath(__file__))

    # For every file in pre-procure, turn it into one in post-procure.
    for content in os.listdir(os.path.join(location, "poll-data/nvidia-smi/pre-procure")):
        if content.split(".")[-1] == ("txt"):
            clean_file(content.split(".")[0], location)


def clean_file(txt_filename, root_dir):
    """
    Turns the named txt file into a csv, cleaning data format and entries as it goes.
    Arguments:
        txt_filename: The name of the hl-smi txt output file to be cleaned.
        root_dir: The path for the parent directory of hl-smi-csvs.
    """
    
    # Open the file for reading, and create the csv header.
    with open(f"{root_dir}/poll-data/nvidia-smi/pre-procure/{txt_filename}.txt") as raw_file:
        lines = raw_file.readlines()
    header = lines[0]

    time_format = "%Y/%m/%d %H:%M:%S.%f"
    first_time = datetime.strptime(lines[1].split(", ")[2], time_format)
    
    # Open file for writing, then write the csv header.
    with open(f"{root_dir}/poll-data/nvidia-smi/post-procure/{txt_filename}.csv", 'w') as outfile:
        # There should be a less brittle way to split this than timestamp.
        outfile.write(f"device,time-diff,{header.split('timestamp, ')[1].replace(', ', ',')}")
        for line in iter(lines):
            if line.__eq__(header):
                # Skip lines that match the header. Since the data is from looping nvidia-smi calls, this is many.
                continue
            vals = line.replace('\n', '').split(", ")
            
            # Format time based on previously-generated formatting.
            time = (datetime.strptime(vals[2], time_format) - first_time).seconds

            # Device id should always be the 2nd val in my calls but this is not necessarily true.
            # Could modify for stability.
            id = vals[1].replace("00000000:", "") # ThetaGPU device names are really messy.
            data = [val.split(' ')[0] for val in vals[3:]]
            # Write in order of id, timestamp (presently idx), all remaining data
            outline = f"{id},{time},{','.join(data)}\n"
            outfile.write(outline)


if __name__ == '__main__':
    main()
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
    for content in os.listdir(os.path.join(location, "theta-csvs/pre-procure")):
        if content.split(".")[-1] == ("txt"):
            clean_file(content.split(".")[0], location)


#TODO increase formatting stability
def clean_file(txt_filename, root_dir):
    
    # Open the file for reading, and create the csv header.
    with open(f"{root_dir}/theta-csvs/pre-procure/{txt_filename}.txt") as raw_file:
        lines = raw_file.readlines()
    header = lines[0]

    time_format = "%Y/%m/%d %H:%M:%S.%f"
    first_time = datetime.strptime(lines[1].split(", ")[2], time_format)
    
    # Open file for writing, then write the csv header
    with open(f"{root_dir}/theta-csvs/post-procure/{txt_filename}.csv", 'w') as outfile:
        # There should be a better way to split this than pci.bus_id
        outfile.write(f"device,time-diff,{header.split('timestamp, ')[1].replace(', ', ',')}")
        for line in iter(lines):
            if line.__eq__(header):
                # Skip lines that match the header. Since the data is from looping nvidia-smi calls, this is many.
                continue
            vals = line.replace('\n', '').split(", ")
            # time = (datetime.strptime(vals[0], time_format).replace(year=cur_year) - first_time).seconds

            time = (datetime.strptime(vals[2], time_format) - first_time).seconds

            # Device id should always be the 2nd val off my calls but this is not necessarily true.
            # Could modify for stability. Also may not be neccesary with single-gpu runs
            id = vals[1].replace("00000000:", "")
            data = [val.split(' ')[0] for val in vals[3:]]
            # Write in order of id, timestamp (presently idx), all remaining data
            outline = f"{id},{time},{','.join(data)}\n"
            outfile.write(outline)


if __name__ == '__main__':
    main()
from datetime import datetime
from mimetypes import init
import os
import numpy as np

def main():
    """
    This file cleans all of the hl-smi generated from Habana at once, located in hl-smi-csvs.
    """

    # Allow script to work when called from anywhere.
    location = os.path.dirname(os.path.abspath(__file__))

    # For every file in pre-procure, turn it into one in post-procure.
    for content in os.listdir(os.path.join(location, "poll-data/hl-smi/pre-procure")):
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
    with open(f"{root_dir}/poll-data/hl-smi/pre-procure/{txt_filename}.txt") as raw_file:
        lines = raw_file.readlines()
    
    # The timestamp in particular is messy, we only need seconds from the start of run.
    header = lines[0]
    time_format = "%a %b %d %H:%M:%S %Z %j"
    first_time = datetime.strptime(lines[1].split(",")[0], time_format)
    cur_year = datetime.now().year

    # Open file for writing, then write the csv header
    # Not particularly robust in regards to device, time-diff, and module_id. Probably could be easily.
    with open(f"{root_dir}/poll-data/hl-smi/post-procure/{txt_filename}.csv", 'w') as outfile:
        # Splitting on module_id is a bit brittle. Should be everything after second col.
        outfile.write(f"device,time-diff,{header.split('module_id, ')[1].replace(', ', ',')}")
        for line in iter(lines):
            # Skip lines that match the header. Since the data is from looping hl-smi calls, this is many.
            if line.__eq__(header):
                continue
            vals = line.replace('\n', '').split(", ")
            
            # Format time based on previously-generated formatting.
            time = (datetime.strptime(vals[0], time_format).replace(year=cur_year) - first_time).seconds

            # Device id should always be the 2nd val in my calls but this is not necessarily true.
            # Could modify for stability.
            id = vals[1]
            data = [val.split(' ')[0] for val in vals[2:]]
            # Write in order of id, timestamp, all remaining data
            outline = f"{id},{time},{','.join(data)}\n"
            outfile.write(outline)


if __name__ == '__main__':
    main()
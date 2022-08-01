from datetime import datetime
from mimetypes import init
import os
import sys
import numpy as np

def main():
    contents = os.listdir("theta-csvs/pre-procure")
    # clean_file('add_mem_data')
    for content in contents:
        if content.split(".")[-1] == ("txt"):
            clean_file(content.split(".")[0])

def clean_file(filename):
    with open(f"theta-csvs/pre-procure/{filename}.txt") as raw_file:
        lines = raw_file.readlines()
    header = lines[0]
    # initial_timestamp = lines[1].split(",")[0]
    # time_format = "%a %b %d %H:%M:%S %Z %j"
    # first_time = datetime.strptime(initial_timestamp, time_format)
    # cur_year = datetime.now().year
    idx = 1
    

    arrays = {i: np.empty((1, len(header.split(","))-1)) for i in range(0, 8)}
    with open(f"theta-csvs/post-procure/{filename}.csv", 'w') as outfile:
        outfile.write(f"bus_id,rough-time,{header.split('pci.bus_id, ')[1].replace(', ', ',')}")
        for line in iter(lines):
            if line.__eq__(header):
                continue
            vals = line.replace('\n', '').split(", ")
            # time = (datetime.strptime(vals[0], time_format).replace(year=cur_year) - first_time).seconds

            id = vals[1]
            data = [val.split(' ')[0] for val in vals[2:]]
            outline = f"{id},{idx},{','.join(data)}\n"
            outfile.write(outline)
            idx += 1





if __name__ == '__main__':
    main()
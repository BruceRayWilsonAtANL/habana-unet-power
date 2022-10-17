import os
import sys
from tabnanny import filename_only

def main():
    """
    This file turns all the run-written txt logs in sub-dir of ./txts (maybe now ./logs)
    into a directory in ./csvs.
    Command line argument: the directory to instrument to be csvs. E.g., python3 txt_to_csv.py habana_init_test
    """
    if len(sys.argv) < 2:
        exit("Usage: python3 txt_to_csv.py <dirname-in-logs>")

    # Allow script to work when called from anywhere
    location = os.path.dirname(os.path.abspath(__file__))

    print(f"location: {location}")

    # Get every .txt in the directory, passing their names w/ other path arguments
    contents = os.listdir(os.path.join(f"{location}/logs", sys.argv[1]))
    for content in contents:
        if content.split(".")[-1] == ("txt"):
            csvify_file(content.split(".")[0], location, sys.argv[1])


def csvify_file(txt_filename, root_dir, group):
    """
    Turns the named txt file into CSV format, writes it out in the equivilent position in csv dir.
    Arguments:
        txt_filename: the name of the file, without .txt or .csv
        root_dir: the shared parent directory of logs and csvs
        group: the dir in which the text file can be found and to place the csv file
    """

    # Create the csv directory if necessary
    if not os.path.exists(f"{root_dir}/csvs/{group}"):
        os.mkdir(f"{root_dir}/csvs/{group}")

    # Get the file, setup loop parameters
    with open(f"{root_dir}/logs/{group}/{txt_filename}.txt") as txt_file:
        lines = txt_file.readlines()
    have_written = False
    writing = False

    # Begin writing
    fileOut = f"{root_dir}/csvs/{group}/{txt_filename}.csv"
    print(f'fileOut: {fileOut}')

    with open(fileOut, 'w') as output:
        for line in lines:
            # Looking for the Begin and End CSV flags to signpost what should be written
            if line.__eq__("Begin CSV\n"):
                # Remove the previous writes if run into a new run.
                # This is to overwrite false starts. Rarely removes actually useful data
                # The command to run the model is also saved, but remains in the text
                if have_written:
                    output.seek(0)
                    output.truncate()
                else:
                    have_written = True
                writing = True
            elif line.__eq__("End CSV\n"):
                writing = False
            elif writing:
                output.write(line)

if __name__ == '__main__':
    main()
import os
import sys

def main():
    """
    This file turns all the run-written txt logs in sub-dir of ./txts into a dir in ./csvs
    command line argument: the directory to instrument to be csvs. E.G. python3 txt_to_csv.py habana02_large_batch
    """

    # Allow script to work when called from anywhere
    location = os.path.dirname(os.path.abspath(__file__))

    # Get every .txt in the directory, passing their names w/ other path arguments
    contents = os.listdir(os.path.join(f"{location}/txts", sys.argv[1]))
    for content in contents:
        if content.split(".")[-1] == ("txt"):
            csvify_file(content.split(".")[0], location, sys.argv[1])
    

def csvify_file(txt_filename, root_dir, group):
    """
    Turns the named txt file into CSV format, writes it out in the equivilent position in csv dir.
    Arguments:
        txt_filename: the name of the file, without .txt or .csv
        root_dir: the shared parent directory of txts and csvs
        group: the dir in which the text file can be found and to place the csv file
    """

    # Create the csv directory if necessary
    if not os.path.exists(f"{root_dir}/csvs/{group}"):
        os.mkdir(f"{root_dir}/csvs/{group}")

    # Get the file, setup loop parameters
    with open(f"{root_dir}/txts/{group}/{txt_filename}.txt") as txt_file:
        lines = txt_file.readlines()
    have_written = False
    writing = False

    # Begin writing
    with open(f"{root_dir}/csvs/{group}/{txt_filename}.csv", 'w') as output:
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
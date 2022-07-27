import os
import sys

def main():
    location = os.path.dirname(os.path.abspath(__file__))
    contents = os.listdir(os.path.join(f"{location}/txts", sys.argv[1]))
    for content in contents:
        if content.split(".")[-1] == ("txt"):
            csvify_file(content.split(".")[0], location, sys.argv[1])
    

def csvify_file(txt_filename, base_dir, parent):
    with open(f"{base_dir}/txts/{parent}/{txt_filename}.txt") as txt_file:
        lines = txt_file.readlines()
    
    have_written = False
    writing = False
    with open(f"{base_dir}/csvs/{parent}/{txt_filename}.csv", 'w') as output:
        for line in lines:
            if line.__eq__("Begin CSV\n"):
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
Outside of running the unet and concurrent profiling, this directory contains the majority of
this project's scripts, as well as the data generated.

Of note, these scripts are completely unnecessary to have on a remote host.
Primarily, the log files should be exported and them scrubbed on a local machine; they're not
particularly large.


## Directory Contents
*  `csvs`:
    The directory containing the runtime-generated data, post-cleaning from `logs`.
    Has different subdirectories for different categories of run, csvs within each.
    

*  `logs`:
    The directory where runtime-generated log files should be stored in sub-directories after sorting.
    Each log file should have the initial python call to note the arguments used, followed by
    lines in the format of a CSV file. There might be multiple false starts (or accidental writes
    to the same file), so only the last complete block of BEGIN CSV - END CSV is stored.
    `txt_to_csv.py` uses these files and outputs the csvs in `csvs`.


*  `pngs`:
    The place where `analysis.py` saves its figures. Can have multiple subdirectories as well.


*  `poll-data`:
    The directory for all the smi-generated data. Split into two subdirectories, `hl-smi` and `nvidia-smi`
    (for the command names) and two subdirectories within those, post and pre-procure for whether the data
    has begun to be cleaned or not. `nvidia_smi.py` and `hl_smi.py` clean those files and move txts from 
    `pre-procure` to `post-procure` as somewhat less noisy csvs.


*  `unsorted-logs`:
    The directory where `unet.py` should be writing out its runtime logs. Move them to the allotted 
    subdirectory in `logs` to use.
    

*  `analysis.py`:
    The primary file to run here. Specifying `run` will get various data about runtime logs (have to
    edit a few variables in the script for this, unfortunately. They're designated toward the top).
    Instead providing the argument `smi` will launch `analysis_smi.py`, which gleans data from the
    device profilers and visualizes that. That script also needs a few variable edits for new files.


*  `analysis_smi.py`:
    See `analysis.py`. Works as-is, mostly, but to capture varying runs and profiles its useful to dig in
    and change which frames are getting visualized and how, which should entirely take place in the 
    main function within.


*  `hl_smi.py`:
    Turns every text log in `poll-data/hl-smi/pre-procure` into a csv in the corresponding `post-procure`.


*  `nvidia_smi.py`:
    `hl_smi.py`, except for `nvidia-smi` data from ThetaGPU instead of Habana. Both scripts are a bit on
    the brittle side, expect to have to change if changing the smi's polling.


*  `txt_to_csv.py`:
    Cleans every text log in a specified directory of `logs`'s, generates and places csvs in a
    corresponding dir in `csvs`. Bit more robust than the other two, has to deal with dirtier data
    thanks to false starts, accidentally shared run names, etc...

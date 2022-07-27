Reminder: various setup commands

```
source /etc/profile.d/habanalabs.sh
source ~/aevard_venv/bin/activate
export HABANA_PROFILE=1
export PYTHON=/home/aevard/aevard_venv/bin/python
export PYTHONPATH=/home/aevard/Model-References:/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
```
`profile.d` for zsh, `activate` and `PYTHON` for any python run and `PYTHONPATH` for references models. `HABANA_LOGS` for `hl-smi` device profiling, `HABANA_PROFILE` for run-specific profiling.

I am also uncertain of ways to move the files from here without using Globus Connect.
For that, I open `/lambda0/homes/aevard/` on Globus connect, and `/~/Documents/Argonne/Habana/` for my mounting.
Then transfer appropriate files.

## Directory Contents
* `unet_bench`:
    Staging location of Habana versions of UNet

* `Model-References`:
    Sample apps from Habana, ver 1.5.0. (.gitignore'd, but should be there)

* `mnist.py`  
`train_fp16_bs16.py`:
    Models that `unet_bench`'s Habana adaptation is based off of.

## Important Scripts
* `unet_bench/unet.py`:
    The main script to execute.

* `scripts/build-hl-smi-csv`:
    The device profiling bash script. Modify for length of runs and output params.

* `performance/analysis.py`:
    The script for all things analysis.

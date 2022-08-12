THIS AND OTHER READMES PRIMARILY USED AS REMINDERS FOR MYSELF. will change that in the next week.
Reminder: various setup commands

```
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
export PYTHONPATH=/home/aevard/Model-References:/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
```


## Directory Contents
* `unet_bench`:
    Staging location of Habana versions of UNet

* `Model-References`:
    Sample apps from Habana, ver 1.5.0. (.gitignore'd, but should be there)

## Important Scripts
* `unet_bench/unet.py`:
    The main script to execute.

* `scripts/build-hl-smi-csv`:
    The device profiling bash script. Modify for length of runs and output params.

* `performance/analysis.py`:
    The script for all things analysis.

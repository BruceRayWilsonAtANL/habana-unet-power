Reminder: until the shell defaulting to zsh is fixed, must use:

```
source /etc/profile.d/habanalabs.sh
export PYTHON=/home/aevard/aevard_venv/bin/python
export PYTHONPATH=/home/aevard/Model-References:/home/aevard/aevard_venv/bin/python
```
before running the script.

I am also uncertain of ways to move the files from here without using Globus Connect.
For that, I open `/lambda_stor/homes/aevard/` on Globus connect, and `/~/Documents/Argonne/Habana/` for my mounting.
Then transfer appropriate files.

## Directory Contents
* `unet_bench`:
    Staging location of Habana versions of UNet

* `Model-References`:
    Sample apps from Habana, ver 1.5.0. (.gitignore'd, but should be there)

* `mnist.py`  
`train_fp16_bs16.py`:
    Models that `unet_bench`'s Habana adaptation is based off of.

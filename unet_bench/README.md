
## Directory Contents

* `weights`:
    The saved weights of models in the `.pt` format. Just to store weights, pretty much.

* `data`:
    The kaggle lgg-mri-segmentation dataset, in all its cached and un-cached forms.

* `models`:
    Versions of the UNet model for `unet.py` to import the model from.

* `performance`:
    Where performance logs are saved, cleaned, and analyzed.

* `scripts`:
    Where a few assorted scripts are stored.

* `dataset.py`:
    File contaning BrainSegmentationDataset. Manages the kaggle_3m dataset.

* `unet_home.py`:
    The basic version of the modified unet that can be run locally. E.G, no Habana-specific scripting.

* `unet.py`:
    The primary unet to run on Habana.

* `unet_local.py`:
    A version of the unet modified to work locally and on theta.

* `utils.py`:
    A utility file carried over from the Habana scripts `unet.py` is based off of.

* `run-commands.txt`:
    The file that houses various commands to run for different versions of the unet model.
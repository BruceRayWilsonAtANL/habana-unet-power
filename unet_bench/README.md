
## Directory Contents

* `weights`:
    The saved weights of models in the `.pt` format. Doesn't matter outside of runs.

* `data`:
    The kaggle lgg-mri-segmentation dataset, in all its cached and un-cached forms.

* `models`:
    Versions of the UNet model for `unet.py` to import the model from.

* `performance`:
    Where performance logs are saved, cleaned, and analyzed.

* `scripts`:
    Where a few assorted scripts are stored.

* `dataset.py`:
    File contaning BrainSegmentationDataset and its helper functions. Manages the kaggle_3m dataset.

* `unet_home.py`:
    The basic version of the modified unet that can be run locally. E.G, no Habana-specific scripting.

* `unet_pet.py`:
    A version of the modified unet that uses the `OxfordIIITPet` dataset from PyTorch. Better to build off of, since no `transforms` or `datasets` overwriting here.

* `unet.py`:
    The primary unet to run on Habana.

* `utils.py`:
    A utility file carried over from the Habana scripts `unet.py` is based off of.



### Commands

(Within Habana)
`$PYTHON unet.py --hpu --use_lazy_mode --epochs 50` is the current default.

Remember to specify a different `--weights-file` than `unet.pt` if running two or more models at once, or if wanting to save the weights file.

This area is set apart to stage various versions runs, so the commands don't need to be written on the remote. Remove the ticks, of course.

`$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-1 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256.pt`

`$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-batch-256-1 --epochs 50 --image-size 128 --batch-size 256 --cache-path kaggle_duped_cache --weights-file 128-b-256-1.pt`

`$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-batch-256-2 --epochs 50 --image-size 128 --batch-size 256 --cache-path kaggle_duped_cache --weights-file 128-b-256-2.pt`

`$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-batch-256-3 --epochs 50 --image-size 128 --batch-size 256 --cache-path kaggle_duped_cache --weights-file 128-b-256-3.pt`

`$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-csv-check --epochs 25 --image-size 32 --cache-path kaggle_cache --weights-file 32-csv.pt`
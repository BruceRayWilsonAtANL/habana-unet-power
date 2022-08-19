
## Directory Contents

* `weights`:
    The saved weights of models in the `.pt` format. Solely to store weights.

* `data`:
    The kaggle lgg-mri-segmentation dataset, in all its cached and un-cached forms.

* `models`:
    Versions of the UNet model for `unet.py` to import the model from. Only one currently, there used to be more.

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
    A version of the unet modified to work locally and on ThetaGPU.

* `utils.py`:
    A utility file carried over from the Habana scripts `unet.py` is based off of.

* `run-commands.txt`:
    The file that houses various commands to run for different versions of the unet model.


## Basic commands for unet.py, and the important flags

Simple Habana unet run:
`$PYTHON unet.py --hpu --use_lazy_mode --run-name basic-habana-1 --image-size 64 --epochs 5 --weights-file b-h-1.pt`


Simple ThetaGPU unet run:
`python3 unet_local.py --run-name basic-theta-1 --image-size 64  --epochs 5 --weights-file b-t-1.pt`
(`scikit-image` may need to be installed into the conda environment first)


Simple multi-card Habana unet run:
`mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-workers-2 --epochs 5 --weights-file 64-t-w-2.pt --world-size 2 --num-workers 2`


First, note that runs of the unet on local machines and ThetaGPU should be done with `unet_local.py`, instead of `unet.py`. The normal one has various imports of Habana architecture-specific modules.

### Unet flags
* `--run-name`: Name of the run, which is used for the name of the runtime log file. NO DEFAULT.

* `--weights-files`: Name of the weights file. Not really important for this, still better to keep from colliding. I use the abbreviation of the run name, anything works. Defaults to unet.pt.

* `--batch-size`, `--test-batch-size`, `--epochs`: self explanitory. defaults to 64, 1 (probably induces runtime issues actually. Should change this), and 25. 25 is usually too many, 5 is often sufficient.

* `--lr`, `--seed`, `--layers`: Nearly unused, still present. Can ignore.

* `--hpu`: Flag to use hpu device. Necessary in all Habana runs, and a Habana specific parameter. Defaults to false.

* `--use_lazy_mode`: [Eager mode](https://docs.habana.ai/en/latest/PyTorch/PyTorch_User_Guide/PyTorch_Gaudi_Integration_Architecture.html#eager-mode) performed much worse in all tested runs, Habana specific parameter. Defaults to false, should be flagged every run (or swapped to true by default).

* `--aug-angle`, `--aug-scale`: Hyperparameters for the dataset. Didn't change much, doesn't really change model performance in regards to metrics tracked, kept anyway. 

* `--image-size`: Flag for what image size the model should expect. DOES NOT ACTUALLY CHANGE THE TRUE IMAGE SIZE OF THE DATASET UNLESS LOADING FROM NON-CACHED DATASET. The next flags will cover that. Defaults to 32, really should default to 64, used that if not higher nearly every run.

* `--im-cache`: Whether to load the data from the raw dataset or a cache. Loading from raw takes upwards of ten minutes due to being taxing and not optimized for Habana, so don't use this flag unless necessary. Defaults to false, so when set it will load a raw set and cache that set.

* `--data-path`: Name of which raw dataset to use from `data`, defaults to `kaggle_3m`. Only necessary to swap when caching a new dataset or image size for said dataset. Common use case is precaching a duplicated dataset for longer runs. My advice is to do an initial run with `epochs=1`, then do the real ones with the `cache-path` set afterwards.

* `--cache-path`: Path of the cache to load/store from `data`, defaults to `kaggle_cache`, which should be image size 64. Example use case is to make or load from a `kaggle_256_cache`.

### Distributed specific flags

* `--distributed`: Self explanitory. Enabling makes the rest of these relevant, mostly.

* `--rank`: The rank of the card, set to 0, mpiruns should be enabled to overwrite this with `--rank-by core`.

* `--world-size`: The number of cards. Should be equal to number of cards in the mpirun's `-n`, defaults to 1.

* `--num-workers`: The number of workers used in the dataloader's instatiation. Defaults to 1, there's code that sets its minimum as the `world-size`.

* `--process-per-node`, `--dist-url`: Present for the original unet code's functionality, didn't really use.


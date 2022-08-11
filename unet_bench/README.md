
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

`$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-duped-128-3 --epochs 50 --image-size 128 --cache-path kaggle_duped_cache --weights-file 128-b-128-3.pt`

`qsub -I -n 1 -t 60 -A datascience -q full-node --attrs enable_ssh=1`
`qsub -I -n 1 -t 60 -A datascience -q single-gpu --attrs enable_ssh=1`
`module load conda/pytorch`
`conda activate`
`pip install --user scikit-image`

`./build-nvidia-smi-csv 64-max-theta.txt &`
`python3 unet.py --run-name 64-max --image-size 64 --batch-size 64 --epochs 2 --cache-path kaggle_max_cache --weights-file 64-m-t.pt`

`python3 unet.py --run-name 128-normal-theta-1 --epochs 50 --image-size 128 --cache-path kaggle_128_cache --weights-file 128-n-t-1.pt`

`mpirun -np 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name 64-duped-cards-2 --epochs 25 --image-size 64 --batch-size 256 --cache-path kaggle_duped_64_cache --weights-file 64-d-c-2.pt --world-size 2`

`$PYTHON unet.py --hpu --use_lazy_mode --run-name distrib-compare --epochs 10 --cache-path kaggle_cache --weights-file d-c-1.pt`
COMMAND WORKS. Currently goal is to see if it scales to 4, whether it repeats images, and that weird error. Then monitor.
Might need to reset the environment variables again.


Command for particularly long run:

`$PYTHON unet.py --hpu --use_lazy_mode --run-name 64-max-1 --image-size 64 --batch-size 64 --epochs 2 --cache-path kaggle_max_cache --weights-file 64-m-1.pt`

`$PYTHON unet.py --hpu --use_lazy_mode --run-name 64-max-2 --image-size 64 --batch-size 64 --epochs 2 --cache-path kaggle_max_cache --weights-file 64-m-2.pt`

`$PYTHON unet.py --hpu --use_lazy_mode --run-name 64-max-3 --image-size 64 --batch-size 64 --epochs 2 --cache-path kaggle_max_cache --weights-file 64-m-3.pt`


`-I` may also be an option for interactive mode. Prepare nvidia-smi command here. Reduce epochs to fit in 60 minutes.

nvidia-smi --query-gpu=gpu_name,gpu_bus_id,utilization.gpu,power.draw --format=csv

reverse the & order I'm thinking
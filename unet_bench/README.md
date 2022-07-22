
## Directory Contents

* `dataset.py`:
    File contaning BrainSegmentationDataset and its helper functions. Manages the kaggle_3m dataset.

* `kaggle_3m`  
  `kaggle_cache`:
    The kaggle lgg-mri-segmentation dataset, and a pre-processed version to load faster.

* `unet_home.py`:
    The basic version of the modified unet that can be run locally. E.G, no Habana-specific scripting.

* `unet_pet.py`:
    A version of the modified unet that uses the `OxfordIIITPet` dataset from PyTorch. Better to build off of, since no `transforms` or `datasets` overwriting here.

* `unet.py`:
    The primary unet to run on Habana.

* `unet.pt`:
    The saved weights of `unet.py`. Not always present.

* `utils.py`:
    A utility file carried over from the Habana scripts `unet.py` is based off of.



### Commands

(Within Habana)
`$PYTHON unet.py --hpu --use_lazy_mode --epochs 50` is the current default.

Remember to specify a different `--weights-file` than `unet.pt` if running two or more models at once, or if wanting to save the weights file.

Current additions are `--im-path` (bool), and `--cache-path` (str). Testing `--image-size` as well.

`$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-1 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256.pt`

`$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-b-32-1 --batch-size 32 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-32-1.pt`

`$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-b-64-1 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-64-1.pt`

`$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-b-128-1 --batch-size 128 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-128-1.pt`

`$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-csv-check --epochs 25 --image-size 32 --cache-path kaggle_cache --weights-file 32-csv.pt`
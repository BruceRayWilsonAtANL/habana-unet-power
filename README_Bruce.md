# Power

## Terminal 1

Open Terminal 1 from dev machine

```bash
ssh -J wilsonb@homes.cels.anl.gov wilsonb@habana-01.ai.alcf.anl.gov
```

Continue

```bash
export PYTHON=/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
source /home/aevard/aevard_venv/bin/activate


cd DL/BruceRayWilsonAtANL/habana-unet-power/unet_bench/

# This only gets done once of course.
cp -r /home/aevard/apps/unet_bench/data/kaggle_duped_cache/ data/kaggle_duped_cache

cd /home/wilsonb/DL/BruceRayWilsonAtANL/habana-unet-power/unet_bench/scripts

# This only gets done once of course.
chmod 755 build-hl-smi-csv
```

Start the power monitoring script with a specified output file.

```bash
./build-hl-smi-csv post-git-test.txt
```

## Terminal 2

Start Terminal 2 from your dev machine

Open a new terminal.

```bash
ssh -J wilsonb@homes.cels.anl.gov wilsonb@habana-01.ai.alcf.anl.gov
```

Continue

```bash
export PYTHON=/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
source /home/aevard/aevard_venv/bin/activate
cd DL/BruceRayWilsonAtANL/habana-unet-power/unet_bench/
```

```bash
time $PYTHON unet.py --hpu --use_lazy_mode --run-name habana-worker-1-duped --epochs 5 --cache-path kaggle_duped_cache --weights-file h-w-1-d.pt
time mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name habana-worker-2-duped --epochs 5 --cache-path kaggle_duped_cache --weights-file h-w-2-d.pt --world-size 2 --num-workers 2
time mpirun -n 4 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name habana-worker-4-duped --epochs 5 --cache-path kaggle_duped_cache --weights-file h-w-4-d.pt --world-size 4 --num-workers 4
```

When the runs have finished switch to terminal 1

## Switch to Terminal 1

<Ctrl+c> out of the power monitoring scripts

```bash
# I don't know what Andre did here.
cat performance/load-size-128.txt | pbcopy
xclip
pbcopy
```

## Copy Files to Development Machine



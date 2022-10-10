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

cd ~/DL/github.com/BruceRayWilsonAtANL/habana-unet-power

# This only gets done once of course.
mkdir data
cp -r /home/aevard/apps/unet_bench/data/kaggle_duped_cache/ data/kaggle_duped_cache

cd ~/DL/github.com/BruceRayWilsonAtANL/habana-unet-power/unet_bench/scripts

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

The argument **run-name** is used to create the output file name.  The Python code is:

```python
LOG_FILE = "performance/unsorted-logs/{}.txt"
# ...
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_FILE.format(args.run_name))
```

**Example:**  The first **run-name** is **habana-worker-1-duped** so, the relative path of the
output file is:

```console
./performance/unsorted-logs/habana-worker-1-duped.txt
```

When the runs have finished switch to terminal 1

## Switch to Terminal 1

<Ctrl+c> out of the power monitoring scripts

```bash
# I don't know what Andre did here.  This just may be a copy to the clipboard.
cat performance/load-size-128.txt | pbcopy
xclip
pbcopy
```

### Use Repo

**NOTE:** Do not add **data/kaggle_duped_cache/**

Use **git add** to add files to the repo.
Use **git commit -am "Added run results."**
Use **git pull**
Use **git push**

## Post Processing

### Git Pull Repo

Use **git pull**

### Run Post-Processing Scripts

#### Venv

```bash
# Activate venv
python3.8 -m venv --system-site-packages ~/venvpower
source ~/venvpower/bin/activate

cd ~/DL/github.com/BruceRayWilsonAtANL/habana-unet-power/unet_bench/performance
```

#### Move Log Files

The log file format should be:

```python
LOG_FILE = "performance/unsorted-logs/{}.txt"
# ...
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_FILE.format(args.run_name))
```

`unet.py` should be writing its runtime logs to **unsorted-logs**.  Move the logs to the **logs** directory.

```bash
cd ~/DL/github.com/BruceRayWilsonAtANL/habana-unet-power/unet_bench/performance
```

Let us say you want to call your batch of runs 'habana_init_test'.

```bash
cd ~/DL/github.com/BruceRayWilsonAtANL/habana-unet-power/unet_bench/performance
mkdir -p logs/habana_init_test

mv unsorted-logs/* logs/habana_init_test
```

#### Process Log Files

If necessary,

```bash
cd /home/bwilson/DL/github.com/BruceRayWilsonAtANL/habana-unet-power/unet_bench/performance
```

Run text to csv conversion script.

```bash
python3 txt_to_csv.py <dirname-in-logs>
```

Example:

```bash
python3 txt_to_csv.py habana_init_test
```

Example output:

```console
location: /home/bwilson/DL/BruceRayWilsonAtANL/habana-unet-power/unet_bench/performance
fileOut: /home/bwilson/DL/BruceRayWilsonAtANL/habana-unet-power/unet_bench/performance/csvs/habana_init_test/habana-worker-1-duped.csv
fileOut: /home/bwilson/DL/BruceRayWilsonAtANL/habana-unet-power/unet_bench/performance/csvs/habana_init_test/habana-worker-4-duped.csv
fileOut: /home/bwilson/DL/BruceRayWilsonAtANL/habana-unet-power/unet_bench/performance/csvs/habana_init_test/habana-worker-2-duped.csv
```

#### Analyze CSV Files

How to get this file created or something simular?

```console
    # FileNotFoundError: [Errno 2] No such file or directory: '/home/bwilson/DL/github.com/BruceRayWilsonAtANL/habana-unet-power/unet_bench/performance/poll-data/hl-smi/post-procure/habana_init_test.csv'
```

```bash
python -m pdb analysis.py smi all
python analysis.py smi all
```

analysis_smi.py
    load_hl_csv()
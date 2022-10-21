# Power

## Terminal 1

Open Terminal 1 from dev machine

```bash
ssh -J wilsonb@homes.cels.anl.gov wilsonb@habana-01.ai.alcf.anl.gov
```

Setup environment

```bash
export PYTHON=/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
source /home/aevard/aevard_venv/bin/activate
cd ~/DL/github.com/BruceRayWilsonAtANL/habana-unet-power
```

Remove old log files.

```bash
cd ~/DL/github.com/BruceRayWilsonAtANL/habana-unet-power/unet_bench/
rm performance/unsorted-logs/*
cd ..
```

This only gets done once of course.

```bash
mkdir data
cd data
cp -r /home/aevard/apps/unet_bench/data/kaggle_duped_cache/ .

cd ~/DL/github.com/BruceRayWilsonAtANL/habana-unet-power/unet_bench/scripts

# This only gets done once of course.
chmod 755 build-hl-smi-csv
```

Start the power monitoring script with a specified output file.

```bash
cd /home/wilsonb/DL/github.com/BruceRayWilsonAtANL/habana-unet-power/unet_bench/scripts
./build-hl-smi-csv unet-test.txt
./build-hl-smi-csv resnet50.txt
```

**NOTE** Remember this file.  cd ~/DL/github.com/BruceRayWilsonAtANL/habana-unet-power/unet_bench/scripts/unet-test.txt.
Below it is referred to as the **device-profiler log**.

## Terminal 2

Start Terminal 2 from your dev machine

Open a new terminal.

```bash
ssh -J wilsonb@homes.cels.anl.gov wilsonb@habana-01.ai.alcf.anl.gov
```

### ResNet50

Setup environment. (From ~/rn50.sh)

```bash
cd /home/wilsonb/DL/Habana/Model-References/TensorFlow/computer_vision/Resnets/resnet_keras/
export PYTHON=$(which python3)
export PYTHONPATH=/home/wilsonb/DL/Habana/Model-References/:$(which python)
export HABANA_LOGS=~/.habana_logs
```

Do runs:

```bash
time $PYTHON resnet_ctl_imagenet_main.py \
    -dt bf16 \
    --data_loader_image_type bf16 \
    -te 5 \
    -bs 256 \
    --optimizer LARS \
    --base_learning_rate 9.5 \
    --warmup_epochs 3 \
    --lr_schedule polynomial \
    --label_smoothing 0.1 \
    --weight_decay 0.0001 \
    --single_l2_loss_op \
    --data_dir /lambda_stor/data/imagenet/tf_records
time mpirun \
    --allow-run-as-root \
    --bind-to core \
    -np 2 \
    --map-by socket:PE=7 \
    --merge-stderr-to-stdout \
       $PYTHON resnet_ctl_imagenet_main.py \
            --dtype bf16 \
            --data_loader_image_type bf16 \
            --use_horovod \
            -te 5 \
            -bs 256 \
            --optimizer LARS \
            --base_learning_rate 9.5 \
            --warmup_epochs 3 \
            --lr_schedule polynomial \
            --label_smoothing 0.1 \
            --weight_decay 0.0001 \
            --single_l2_loss_op \
            --data_dir /lambda_stor/data/imagenet/tf_records
time mpirun \
    --allow-run-as-root \
    --bind-to core \
    -np 4 \
    --map-by socket:PE=7 \
    --merge-stderr-to-stdout \
       $PYTHON resnet_ctl_imagenet_main.py \
            --dtype bf16 \
            --data_loader_image_type bf16 \
            --use_horovod \
            -te 5 \
            -bs 256 \
            --optimizer LARS \
            --base_learning_rate 9.5 \
            --warmup_epochs 3 \
            --lr_schedule polynomial \
            --label_smoothing 0.1 \
            --weight_decay 0.0001 \
            --single_l2_loss_op \
            --data_dir /lambda_stor/data/imagenet/tf_records
time mpirun \
    --allow-run-as-root \
    --bind-to core \
    -np 8 \
    --map-by socket:PE=7 \
    --merge-stderr-to-stdout \
       $PYTHON resnet_ctl_imagenet_main.py \
            --dtype bf16 \
            --data_loader_image_type bf16 \
            --use_horovod \
            -te 5 \
            -bs 256 \
            --optimizer LARS \
            --base_learning_rate 9.5 \
            --warmup_epochs 3 \
            --lr_schedule polynomial \
            --label_smoothing 0.1 \
            --weight_decay 0.0001 \
            --single_l2_loss_op \
            --data_dir /lambda_stor/data/imagenet/tf_records
time mpirun -v \
    --allow-run-as-root \
    -np 16 \
    --mca btl_tcp_if_include 192.168.201.0/24 \
    --tag-output \
    --merge-stderr-to-stdout \
    --prefix /usr/local/openmpi \
    -H 140.221.77.101:8,140.221.77.102:8 \
    -x GC_KERNEL_PATH \
    -x HABANA_LOGS \
    -x PYTHONPATH \
    -x HCCL_SOCKET_IFNAME=enp75s0f0,ens1f0 \
    -x HCCL_OVER_TCP=1 \
    $PYTHON resnet_ctl_imagenet_main.py \
        -dt bf16 \
        -dlit bf16 \
        -bs 256 \
        -te 5 \
        --use_horovod \
        --data_dir /lambda_stor/data/imagenet/tf_records/ \
        --optimizer LARS \
        --base_learning_rate 9.5 \
        --warmup_epochs 3 \
        --lr_schedule polynomial \
        --label_smoothing 0.1 \
        --weight_decay 0.0001 \
        --single_l2_loss_op \
        --model_dir=`mktemp -d`
```

### UNet

```bash
export PYTHON=/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
source /home/aevard/aevard_venv/bin/activate
cd ~/DL/github.com/BruceRayWilsonAtANL/habana-unet-power/unet_bench/
```

```bash
time $PYTHON unet.py --hpu --use_lazy_mode --run-name habana-worker-1-duped --epochs 5 --cache-path kaggle_duped_cache --weights-file h-w-1-d.pt
time mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name habana-worker-2-duped --epochs 5 --cache-path kaggle_duped_cache --weights-file h-w-2-d.pt --world-size 2 --num-workers 2
time mpirun -n 4 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name habana-worker-4-duped --epochs 5 --cache-path kaggle_duped_cache --weights-file h-w-4-d.pt --world-size 4 --num-workers 4
time mpirun -n 8 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name habana-worker-8-duped --epochs 5 --cache-path kaggle_duped_cache --weights-file h-w-8-d.pt --world-size 8 --num-workers 8
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

### Wrap up

When the runs have finished switch to terminal 1

## Switch to Terminal 1

<Ctrl+c> out of the power monitoring scripts

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

```text
Once the device profiler was running, I set the unet(s) to run. Information is printed
on their shells, and everything that is printed should also be logged, and more than that.
The location of the logging is in `performance/unsorted-logs`. These files and the
device profiling logs all need to be extracted from the remote in whatever manner
applicable. I genereally used Globus. - Alex
I have this done and documented below. -BRW
```

The log file format should be:

```python
LOG_FILE = "performance/unsorted-logs/{}.txt"
# ...
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_FILE.format(args.run_name))
```

`unet.py` should be writing its runtime logs to **unsorted-logs**.  Move the logs to the **logs** directory.

```bash
cd ~/DL/github.com/BruceRayWilsonAtANL/habana-unet-power/unet_bench/performance
mv unsorted-logs/* logs/habana_init_test
```

```bash
ls -al logs/habana_init_test
```

The output should look like:

```console
total 28
drwxrwxr-x 2 bwilson bwilson 4096 Oct 11 08:53 .
drwxrwxr-x 5 bwilson bwilson 4096 Oct 10 10:44 ..
-rw-rw-r-- 1 bwilson bwilson  939 Oct 11 08:02 habana-worker-1-duped.txt
-rw-rw-r-- 1 bwilson bwilson 1372 Oct 11 08:02 habana-worker-2-duped.txt
-rw-rw-r-- 1 bwilson bwilson 2370 Oct 11 08:02 habana-worker-4-duped.txt
-rw-rw-r-- 1 bwilson bwilson 4392 Oct 11 08:02 habana-worker-8-duped.txt
```

```text
The device-profiler log should be placed in
`performance/poll-data/<node-type>/pre-procure`
where node-type is [hl-smi | nvidia-smi].
```

Continuing with the Habana example:

```bash
#
#
cd ~/DL/github.com/BruceRayWilsonAtANL/habana-unet-power/unet_bench/
cp ./scripts/unet-test.txt ./performance/poll-data/hl-smi/pre-procure
cp ./scripts/resnet50*.txt ./performance/poll-data/hl-smi/pre-procure
#
#
```

#### Preprocess Log Files

Preprocess/filter the log file.

```text
Place the runtime logs into
`performance/logs/<name-of-test-category>` to be used. Once placed, I used
combinations of `hl_smi.py`, `nvidia_smi.py`, and `txt_to_csv.py` to clean them. - Andre
```

Let us say you want to call your batch of runs 'habana_init_test'.

```bash
#
#
x
x
x
x

cd ~/DL/github.com/BruceRayWilsonAtANL/habana-unet-power/unet_bench/performance
mkdir -p logs/habana_init_test
mkdir -p logs/resnet_test
#
#
```

For Habana, there are two choices for cleaning data.  They are:

1. hl_smi.py
2. txt_to_csv.py

The second is more robust per Andre.  I used **hl_smi.py**.

##### hl_smi.py

```bash
#
#
cd ~/DL/github.com/BruceRayWilsonAtANL/habana-unet-power/unet_bench/performance

#poll-data/hl-smi/pre-procure/*.txt # Input
#poll-data/hl-smi/post-procure/*.csv # Output
python3 hl_smi.py
#
#
```

#### Analyze Log Files

```text
Then use `analysis.py` and `analysis_smi.py` to extract insights from them.
See `performance/README.md` for more details there, they unfortunately
need to be edited to be used effective, apologies. Generally speaking, though,
I would get the runtime performance as a printed json object from calling
`python3 analysis.py run`, and a figure from `python3 analysis.py smi`.
A png should also get generated in `performance/pngs/<project-name>`. - Andre
```

#### Analyze CSV Files

I am using:

```bash
source ~/venvpower/bin/activate
cd ~/DL/github.com/BruceRayWilsonAtANL/habana-unet-power/unet_bench/performance
python analysis.py smi all
python analysis.py smi model
```

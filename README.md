# Unet Power

Reminder: various setup commands on Habana:

```bash
export HABANA_LOGS=~/.habana_logs
source /home/aevard/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
export PYTHONPATH=/home/bwilson/DL/Habana/Model-References:/home/aevard/aevard_venv/bin/python
```

And setup commands on ThetaGPU:

```bash
qsub -I -n 1 -t 60 -A datascience -q single-gpu --attrs enable_ssh=1
qsub -I -n 1 -t 120 -A datascience -q full-node --attrs enable_ssh=1
module load conda/pytorch
conda activate
pip install --user scikit-image
```

## Directory Contents

* `unet_bench`:
    Staging location of Habana versions of UNet

* `Model-References`:
    Sample apps from Habana. (.gitignore'd, but useful to have around)

## Important Scripts

* `unet_bench/unet.py`:
    The main script to execute.

* `scripts/build-hl-smi-csv`:
    The device profiling bash script. Modify for length of runs and output params.

* `performance/analysis.py`:
    The script for all things analysis.

### Typical process

(Speaking as if the shell is already located in `unet_bench`)
Most commonly, once all relevant scripts had been transfered to one/both of the node(s),
I would stage the commands I wanted to use in `run-commands.txt` (the local one,
doesn't matter one way or the other if on the remote). I would duplicate and/or
cache the dataset if necessary, or transfer it from a node that has it, and then
move to prepare the device profiler.

The device profiler is `scripts/build-hl-smi-csv` or `scripts/build/nvidia-smi-csv`,
depending on the node. Either way, it needs to run concurrently (a script that manages
that all would be a major qol improvement). On Habana this meant an additional shell
running it, on Theta it was ran in the background with &. It's difficult to tell when
exactly the model will end, so I just ensured it was running before & after the model
was running, usually stopping with ^C. It's safe to do that, and also to cancel and
restart for excessively long runs. This command needs one argument for the name of the
file to generate (desired extension is also necessary to specify).

Once the device profiler was running, I set the unet(s) to run. Information is printed
on their shells, and everything that is printed should also be logged, and more than that.
The location of the logging is in `performance/unsorted-logs`. These files and the
device profiling logs all need to be extracted from the remote in whatever manner
applicable. I genereally used Globus.

From there, the device-profiler log should be placed in
`performance/poll-data/<node-type>/pre-process`, while the runtime logs into
`performance/logs/<name-of-test-category>` to be used. Once placed, I used
combinations of `hl_smi.py`, `nvidia_smi.py`, and `txt_to_csv.py` to clean them,
before using `analysis.py` and `analysis_smi.py` to extract insights from them.
See `performance/README.md` for more details there, they unfortunately
need to be edited to be used effective, apologies. Generally speaking, though,
I would get the runtime performance as a printed json object from calling
`python3 analysis.py run`, and a figure from `python3 analysis.py smi`.
A png should also get generated in `performance/pngs/<project-name>`.

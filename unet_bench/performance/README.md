## Contents

* `csvs`:
    Stores the logged performance data in csv format to use.

* `txts`:
    Stores the performance data the model has logged.

* `hl_smi_csvs`:
    Stores the `hl-smi`-generated txts and resulting csvs.

* `hl_smi_csv.py`:
    Creates csvs from txts for the run's own performance statements. Requires being targeted at a directory within `txts` that exists in `csvs`.

* `txt_to_csv.py`:
    Turns every `.txt` in `pre-procure` into a `.csv` in `post-procure`.

* `analysis.py`:
    The primary analysis script. There's a variety of things that need to be modified on a run-by-run process, so dig into it and look for a ## that marks variables to change.

Note that pretty much all of the txts need to be moved off Habana via Globus to be used.
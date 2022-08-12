## Contents

* `build-hl-smi-csv`:
    The bash script that runs `hl-smi` while the models are running. Does not automatically start or stop, must be managed concurrently.

* `build-nvidia-smi-csv`:
    The bash script that runs `nvidia-smi` while the models are running. Does not automatically start or stop, must be managed concurrently in the background.
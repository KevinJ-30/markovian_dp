# Learning Privately from Graphs: Privacy Amplification via Structured Subsampling

This repository contains the code for *Learning Privately from Graphs: Privacy Amplification via Structured Subsampling*, including SparseGNN, baseline methods, and the training and numerical experiments.

## Installation

Run the following commands from this directory to create and activate the environment:

```bash
conda env create -f environment.yml
conda activate graph-subsampling
python -m pip check
```

The environment targets Linux x86_64 and includes the dependencies for all methods. GPU experiments require an NVIDIA driver compatible with CUDA 13.2.

## Datasets

OGB, Twitch, Facebook, and MAG datasets download automatically when an experiment first loads them. Downloads and processed data are cached under `data/`, so the first run needs an internet connection and may take longer.

The GraphSAINT datasets require a manual download. Download the Reddit, Yelp, and Amazon folders from the [GraphSAINT dataset collection](https://drive.google.com/open?id=1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz), linked from the [GraphSAINT repository](https://github.com/GraphSAINT/GraphSAINT#datasets), and extract them into this layout:

```text
data/graphsaint/
    reddit/
    yelp/
    amazon/
```

Each dataset folder should contain `adj_full.npz`, `adj_train.npz`, `feats.npy`, `role.json`, and `class_map.json`. If the download is split across several archives, extract all parts into the same dataset folder. The experiment configurations refer to these datasets as `saint-reddit`, `saint-yelp`, and `saint-amazon`.

You can store the GraphSAINT folders elsewhere by setting `GRAPHSAINT_DATA_ROOT` to their parent directory before running an experiment.

## Running the experiments

The [reproduction guide](reproduce.md) gives the commands for training, repeated runs, result tables, ablations, and numerical figures. Experiment settings are stored in `configs/`.

To preview the frozen validation-selected configurations and run their five-seed repeats on GPU 0:

```bash
python scripts/run_experiments.py configs/main_r1_eps1258_repeats.json --gpus 0 --dry-run
python scripts/run_experiments.py configs/main_r1_eps1258_repeats.json --gpus 0
```

Replace `0` with the GPUs you want to use, such as `0,1`. Results and logs are written to `results/main_r1_eps1258_repeats/`. To continue an interrupted run, use the same command with `--resume`. The reproduction guide also covers fresh tuning. Superseded configurations are retained locally in the Git-ignored `old_configs/` directory.

We also note that the configuration files use a custom-written scheduler, which attempts to add as many jobs as possible to each GPU, since the jobs are largely CPU-dependent. 

## Generated files and version control

Keep experiment configs, source code, tests, documentation, and required
vendored assets in Git. Generated run outputs belong in `results/`, summaries
in `reports/`, and figures in `figures/` or `numerics/figures/`. These complete
directories are ignored, including CSV/JSON metadata, logs, and partial files.
Use these locations for custom output paths too.

Downloaded data and split caches under `data/`, scratch files under `tmp/`,
checkpoints, and Python/test caches are also ignored. CSV/JSON, image/PDF,
HDF5, and notebook files outside generated directories remain trackable so
configs, fixtures, and documentation assets are not silently excluded.
This includes repeat-run configs generated into `configs/`.

Ignore rules do not untrack files already committed or remove them from Git
history. Existing tracked artifacts require a separate index cleanup; changing
`.gitignore` does not delete local results or modify the index.

## Code and tests

The `src/` directory contains the models, sampling, training, and privacy accounting code. The `scripts/` directory contains experiment runners and result summaries, while `numerics/` contains the numerical privacy experiments. See [the script guide](scripts/README.md) for additional command-line options.

To run the tests:

```bash
python -m pytest tests/
```

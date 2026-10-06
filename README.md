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

Replace `0` with the GPUs you want to use, such as `0,1`. Results and logs are written to `results/main_r1_eps1258_repeats/`. To continue an interrupted run, use the same command with `--resume`. The reproduction guide also covers fresh tuning.

We also note that the configuration files use a custom-written scheduler, which attempts to add as many jobs as possible to each GPU, since the jobs are largely CPU-dependent. 

## Code and tests

The `src/` directory contains the models, sampling, training, and privacy accounting code. The `scripts/` directory contains experiment runners and result summaries, while `numerics/` contains the numerical privacy experiments. See the [reproduction guide](reproduce.md#runner-usage) for runner conventions and use each script's `--help` for command-line options.

Training uses `scripts/run_experiments.py` for configured studies and `scripts/run_experiment.py` for individual runs. The worker resolves dataset task metadata through `src/data/task_metadata.py`, performs privacy calibration/accounting during private runs, and invokes the shared training implementations directly; ProGAP runs through its retained partition adapter.

SparseExpand and SparseGNN training support only incoming expansion (Algorithm 5), preserving arc orientation toward each root and capping sampled incoming arcs at 20, 10, then 5 per expanded node by hop. Their APIs have no `direction` option. The shared CSR adjacency builder still supports outgoing storage for DP-GNN; outgoing adjacency cannot be passed to SparseExpand.

To run the tests:

```bash
python -m pytest tests/
```

ProGAP tests use the same Python interpreter as pytest; no separate ProGAP test environment is needed. Binary AUROC scoring is shared through `src.models.objectives._binary_auroc`.

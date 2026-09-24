"""Feature-only MLP on a RelBench entity task; also writes info.json (delta).

BaselineTrainer supervises every node, so it gets labelled entities only.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.datasets import load_dataset                      # noqa: E402
from src.processing.splits import GraphPartition, InductiveSplit  # noqa: E402
from src.training.baselines import BaselineConfig, BaselineTrainer  # noqa: E402


def labelled_partition(data, mask):
    node_ids = torch.where(mask)[0]
    part = Data(x=data.x[node_ids], y=data.y[node_ids],
                edge_index=torch.zeros((2, 0), dtype=torch.long))
    part.num_nodes = int(node_ids.numel())
    return GraphPartition(part, node_ids, {"num_nodes": part.num_nodes})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--hubs", default="drop", choices=["drop", "replicate"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dataset, data = load_dataset(args.dataset, device="cpu",
                                 root="entity", hubs=args.hubs)
    regression = "REGRESSION" in str(dataset.task_type).upper()
    binary = "BINARY" in str(dataset.task_type).upper()
    info = {
        "dataset": args.dataset,
        "task_type": str(dataset.task_type),
        "n_train_rows": int(data.train_mask.sum()),
        "n_val_rows": int(data.val_mask.sum()),
        "n_test_rows": int(data.test_mask.sum()),
        "n_train_nodes": int(data.n_train_nodes),
        "n_nodes": int(data.num_nodes),
        "target_std": float(getattr(dataset, "target_std", 1.0)),
    }
    info["delta"] = 1.0 / info["n_train_nodes"]
    with open(os.path.join(args.out_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))

    split = InductiveSplit(
        train=labelled_partition(data, data.train_mask),
        val=labelled_partition(data, data.val_mask),
        test=labelled_partition(data, data.test_mask),
        masks={"train": data.train_mask, "val": data.val_mask, "test": data.test_mask},
        num_classes=dataset.num_classes,
        path=Path(args.out_dir),
        primary_metric="mae" if regression else "auroc" if binary else "accuracy",
        binary=binary,
    )
    config = BaselineConfig(method="mlp", hidden_size=args.hidden,
                            epochs=args.epochs, batch_size=args.batch_size,
                            seed=args.seed, regression=regression, binary=binary)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = BaselineTrainer(config, device=device).fit(split)
    if regression:
        result["validation_mae"] = result.pop("validation_accuracy")
        result["test_mae"] = result.pop("test_accuracy")
        result.pop("validation_macro_f1")
        result.pop("test_macro_f1")
    with open(os.path.join(args.out_dir, "mlp_result.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    keys = [k for k in result if k.startswith(("validation_", "test_"))]
    print({k: result[k] for k in keys})


if __name__ == "__main__":
    main()

"""Independent workers must not serialize read-only graph preparation."""
import json
import multiprocessing
from pathlib import Path


def _train_at_preprocessing_barrier(root, seed, barrier):
    from types import SimpleNamespace
    from unittest.mock import patch
    import torch
    from torch_geometric.data import Data
    from scripts import run_experiment as worker
    from src.data import datasets
    from src.processing import graphs

    torch.set_num_threads(1)
    nodes = torch.arange(40)
    graph = Data(x=torch.arange(160).reshape(40, 4).float() / 160,
                 y=nodes % 2, edge_index=torch.stack((nodes, (nodes + 1) % 40)), num_nodes=40)
    dataset = SimpleNamespace(task_type="MULTICLASS", primary_metric="accuracy", num_classes=2)
    preprocess = graphs.preprocess_inductive_split

    def concurrent_preprocess(split):
        # Both real workers must reach reconstruction before either can train.
        # Holding a per-dataset lock across this step deadlocks at the barrier.
        barrier.wait(timeout=30)
        return preprocess(split)

    root = Path(root)
    with patch.object(datasets, "load_dataset", side_effect=lambda *a, **kw: (dataset, graph.clone())), \
         patch.object(graphs, "preprocess_inductive_split", side_effect=concurrent_preprocess):
        status = worker.main([
            "--dataset", "fixture", "--method", "mlp", "--lr", "0.01",
            "--batch-size", "8", "--epochs", "1", "--mlp-hidden", "4",
            "--seed", str(seed), "--bootstrap-resamples", "0", "--device", "cpu",
            "--split-root", str(root / "splits"), "--out-dir", str(root / f"seed{seed}"),
        ])
    assert status == 0


def test_workers_prepare_same_dataset_concurrently(tmp_path):
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    processes = [context.Process(target=_train_at_preprocessing_barrier,
                                 args=(str(tmp_path), seed, barrier)) for seed in (0, 1)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=60)
        assert [p.exitcode for p in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
    results = [json.loads((tmp_path / f"seed{seed}" / "result.json").read_text()) for seed in (0, 1)]
    assert {r["seed"] for r in results} == {0, 1}
    assert results[0]["split_file"] == results[1]["split_file"]
    assert all(r["completed_epochs"] == 1 for r in results)

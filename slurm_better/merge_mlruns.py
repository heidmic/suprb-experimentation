#!/usr/bin/env python3
"""
Merge per-worker mlflow local file stores into one shared store.

Each eval worker writes to its own local scratch mlruns/ (to avoid NFS lock
contention). 

Usage:
    python merge_mlruns.py --dest file:///home/user/suprb-experimentation/mlruns \
        --sources /home/user/mlruns-eval-8564/task-0/mlruns \
                  /home/user/mlruns-eval-8564/task-1/mlruns \
                  ...
"""

import argparse
import shutil
from pathlib import Path

import mlflow
from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient


def merge_experiment(
    src_client: MlflowClient,
    dst_client: MlflowClient,
    src_exp,
    dest_exp_id: str,
    src_mlruns_path: Path,
):
    runs = src_client.search_runs(
        experiment_ids=[src_exp.experiment_id],
        max_results=50000,
    )
    print(f"  {len(runs)} runs found in source experiment '{src_exp.name}' ({src_exp.experiment_id})")

    for run in runs:
        info, data = run.info, run.data

        run_name = data.tags.get("mlflow.runName")

        create_tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}

        if run_name is not None:
            create_tags["mlflow.runName"] = run_name

        new_run = dst_client.create_run(
            experiment_id=dest_exp_id,
            start_time=info.start_time,
            tags=create_tags,
        )
        new_run_id = new_run.info.run_id

        # Params
        params = [Param(k, v) for k, v in data.params.items()]

        # Metrics: fetch full history
        metrics = []
        for key in data.metrics:
            for m in src_client.get_metric_history(info.run_id, key):
                metrics.append(Metric(m.key, m.value, m.timestamp, m.step))

        # Tags (skip mlflow internal ones)
        tags = [RunTag(k, v) for k, v in data.tags.items() if not k.startswith("mlflow.") and k != "mlflow.runName"]

        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        for m_chunk in chunks(metrics, 900):
            dst_client.log_batch(new_run_id, metrics=m_chunk, params=[], tags=[])
        if params:
            dst_client.log_batch(new_run_id, metrics=[], params=params, tags=[])
        if tags:
            dst_client.log_batch(new_run_id, metrics=[], params=[], tags=tags)

        # -----------------------------------------------------------------
        # Artifacts: copy directly from the local source mlruns folder
        # Path: <src_mlruns_path>/<experiment_id>/<run_id>/artifacts
        artifacts_src = src_mlruns_path / src_exp.experiment_id / info.run_id / "artifacts"
        if artifacts_src.exists() and any(artifacts_src.iterdir()):
            dst_client.log_artifacts(new_run_id, str(artifacts_src))
        # -----------------------------------------------------------------

        dst_client.set_terminated(new_run_id, status=info.status, end_time=info.end_time)

    print(f"  merged {len(runs)} runs -> {dest_exp_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True, help="Destination tracking URI, e.g. file:///path/to/mlruns")
    parser.add_argument("--sources", nargs="+", required=True, help="One or more source mlruns directories")
    parser.add_argument("--delete-sources", action="store_true", help="Remove source dirs after a successful merge")
    args = parser.parse_args()

    dst_client = MlflowClient(tracking_uri=args.dest)

    for src_dir in args.sources:
        src_path = Path(src_dir)
        if not src_path.is_dir():
            print(f"skip: {src_dir} does not exist")
            continue

        src_uri = f"file://{src_path.resolve()}"
        src_client = MlflowClient(tracking_uri=src_uri)

        print(f"Merging {src_uri} -> {args.dest}")
        for exp in src_client.search_experiments():
            if exp.name == "Default":
                continue

            dest_exp = dst_client.get_experiment_by_name(exp.name)
            if dest_exp is None:
                dest_exp_id = dst_client.create_experiment(exp.name)
            else:
                dest_exp_id = dest_exp.experiment_id

            merge_experiment(src_client, dst_client, exp, dest_exp_id, src_path)

        if args.delete_sources:
            shutil.rmtree(src_path)
            print(f"  removed {src_path}")

    print("Merge complete.")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tabarena.benchmark.experiment import ExperimentBatchRunner, AGModelExperiment
from tabarena.benchmark.models.ag.nanotabpfn.nanotabpfn_model import NanoTabPFNModel
from tabarena.nips2025_utils.end_to_end import EndToEnd
from tabarena.nips2025_utils.tabarena_context import TabArenaContext


if __name__ == '__main__':
    expname = str(Path(__file__).parent / "experiments" / "nanotabpfn")
    leaderboard_dir = str(Path(__file__).parent / "leaderboards" / "nanotabpfn")
    ignore_cache = False  # set to True to overwrite existing caches and re-run from scratch

    ta_context = TabArenaContext()
    task_metadata = ta_context.task_metadata.copy()

    # Binary classification datasets only, capped at threshold of 2,499 rows (8 datasets)
    # to stay within NanoTabPFN's memory budget (attention is O(n_rows ^ 2))
    binary_metadata = task_metadata[
        (task_metadata["NumberOfClasses"] == 2.0)
        & (task_metadata["NumberOfInstances"] < 2_500)
    ]
    datasets = list(binary_metadata["name"])
    folds = list(range(3))
    repeats = list(range(10))  # Datasets with >= 2500 rows only have 3 repeats available

    # Some logging before running the experiments
    import os
    import openml
    openml.config.cache_directory = f"/cs/data/people/{os.getenv('USER')}/.cache/openml"
    openml_cache = Path(openml.config.get_cache_directory())
    print(f"OpenML cache directory: {openml_cache}")
    print(f"\nDatasets to run ({len(datasets)} total):")
    print(f"  {'name':45} {'rows':>8} {'features':>9}")
    print(f"  {'-'*45} {'-'*8} {'-'*9}")
    for _, row in binary_metadata.sort_values("NumberOfInstances").iterrows():
        print(f"  {row['name']:45} {int(row['NumberOfInstances']):>8,} {int(row['NumberOfFeatures']):>9,}")
    print()

    # Defines the evaluations to run and runs them
    methods = [
        AGModelExperiment(
            name="NanoTabPFN-lam0.0",
            model_cls=NanoTabPFNModel,
            model_hyperparameters={"lam": 0.0},
        ),
        AGModelExperiment(
            name="NanoTabPFN-lam0.01",
            model_cls=NanoTabPFNModel,
            model_hyperparameters={"lam": 0.01},
        ),
        AGModelExperiment(
            name="NanoTabPFN-lam0.1",
            model_cls=NanoTabPFNModel,
            model_hyperparameters={"lam": 0.1},
        ),
    ]
    exp_batch_runner = ExperimentBatchRunner(expname=expname, task_metadata=binary_metadata)
    results_lst: list[dict[str, Any]] = exp_batch_runner.run(
        datasets=datasets,
        folds=folds,
        repeats=repeats,
        methods=methods,
        ignore_cache=ignore_cache,
        raise_on_failure=False,
    )

    # Processes results
    end_to_end = EndToEnd.from_raw(results_lst=results_lst, task_metadata=binary_metadata, cache=False, cache_raw=False)
    end_to_end_results = end_to_end.to_results()
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Results:\n{end_to_end_results.model_results.head(100)}")

    # Makes leaderboard, to compare to other models
    leaderboard: pd.DataFrame = end_to_end_results.compare_on_tabarena(
        output_dir=leaderboard_dir,
        subset="classification",
        only_valid_tasks=True,
        plot_with_baselines=True,
    )
    print(f"Leaderboard:")
    print(leaderboard.to_markdown(index=False))
    print(f"View saved figures in {leaderboard_dir}")

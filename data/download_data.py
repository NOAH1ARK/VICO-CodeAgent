# Copyright 2026 Yiming. Licensed under the Apache License, Version 2.0.
# Derived from RLAnything / Open-AgentRL (https://github.com/Gen-Verse/Open-AgentRL),
# Apache-2.0. Modified in this repository; see NOTICE for a summary of changes.
import argparse
from huggingface_hub import hf_hub_download
import shutil


parser = argparse.ArgumentParser(description="Download a dataset from HF hub")
parser.add_argument(
    "--dataset",
    choices=["CodeContests_train","LiveBench-ReasonFlux"],
    required=True,
    help="Which dataset to download"
)
args = parser.parse_args()
dataset = args.dataset


if dataset == "CodeContests_train":
    split = "train"
else:
    split = "test"


cached_path = hf_hub_download(
    repo_id=f"Gen-Verse/{dataset}",
    repo_type="dataset",
    filename=f"{split}/{dataset}.json"
)
shutil.copy(cached_path, f"./{dataset}.json")

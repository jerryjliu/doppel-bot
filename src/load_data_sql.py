# from typing import Iterable, Any
import json

from modal import Dict, Image, Secret, Retries
from .common import (
    stub,
    VOL_MOUNT_PATH,
    output_vol,
    # user_data_path,
    # SLACK_SECRET_NAME
)
from .common_sql import user_data_path
from pathlib import Path

@stub.function(
    retries=Retries(
        max_retries=3,
        initial_delay=5.0,
        backoff_coefficient=2.0,
    ),
    timeout=60 * 60 * 2,
    network_file_systems={VOL_MOUNT_PATH.as_posix(): output_vol},
    cloud="gcp",
)
def load_data_sql(user: str):
    from datasets import load_dataset

    dataset = load_dataset("b-mc2/sql-create-context")

    dataset_splits = {"train": dataset["train"]}
    out_path = user_data_path(user)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    for key, ds in dataset_splits.items():
        with open(out_path, "w") as f:
            for item in ds:
                newitem = {
                    "user": str(user),
                    "input": item["question"],
                    "context": item["context"],
                    "output": item["answer"],
                }
                f.write(json.dumps(newitem) + "\n")
    
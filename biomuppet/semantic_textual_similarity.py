import json
from functools import partial
from pathlib import Path
from typing import List

import datasets
import numpy as np
from bigbio.utils.constants import Tasks
from datasets import DatasetDict
from tqdm import tqdm
from bisect import bisect

from biomuppet.classification import get_classification_meta
from biomuppet.utils import SingleDataset, get_all_dataloaders_for_task, DEBUG


def get_percentile_bins(dataset):
    labels = []
    for label in dataset['train']['label']:
        labels.append(float(label))
    np.bincount(labels)

    percentile_bins = []
    prev_percentile = None
    for i in range(1, 10):
        percentile = np.percentile(labels, q=i*10)
        if percentile != prev_percentile:
            percentile_bins.append(percentile)
        prev_percentile = percentile

    return percentile_bins


def regression_to_classification_bin(example, bins):
    example['labels'] = [bisect(bins, x=float(example['label']))]

    return example


def get_all_sts_datasets() -> List[SingleDataset]:
    sts_datasets = []

    for dataset_loader in tqdm(
            get_all_dataloaders_for_task(Tasks.SEMANTIC_SIMILARITY),
            desc="Preparing STS datasets",
    ):
        dataset_name = Path(dataset_loader).with_suffix("").name

        if "pubhealth" in dataset_name: # not sts but classification fixed in #545 (not yet merged)
            continue
        elif "bio_simlex" in dataset_name: # label contains \n fixed in #541 (not yet merged)
            continue
        elif "umnsrs" in dataset_name: # mayor flaws (not downloadable) fixed in #538 (not yet merged)
            continue



        dataset = datasets.load_dataset(
            str(dataset_loader), name=f"{dataset_name}_bigbio_pairs"
        )

        def replace_(example):
            example["text_1"] = example["text_1"].strip().replace("\t", " ").replace("\n", " ")
            example["text_2"] = example["text_2"].strip().replace("\t", " ").replace("\n", " ")
            return example

        def label_asserts_(example):

            if dataset_name == "mqp":
                label = int(example["label"])
            else:
                label = float(example["label"])

            assert label >= 0, "Oh no"

            # all datasets are range, except mqp (0/1)
            if dataset_name == "biosses":
                assert 0 <= label <= 4, f"Oh no {dataset_name} {example['label']}"
            elif dataset_name in ["bio_simlex", "bio_sim_verb"]:
                assert 0 <= label <= 10, f"Oh no {dataset_name} {example['label']}"
            elif dataset_name == "ehr_rel":
                assert 0 <= label <= 3, f"Oh no {dataset_name} {example['label']}"
            elif dataset_name in ["minimayosrs", "mayosrs"]:
                assert 1 <= label <= 10, f"Oh no {dataset_name} {example['label']}"
            elif dataset_name == "mqp":
                assert label == 0 or label== 1, f"Oh no {dataset_name} {example['label']}"
            elif dataset_name == "umnsrs":
                assert 0 <= label <= 1600, f"Oh no {dataset_name} {example['label']}"

        dataset = dataset.map(replace_)
        dataset = dataset.map(label_asserts_)
        dataset = dataset.remove_columns(['id', "document_id"])

        bins = get_percentile_bins(dataset)
        dataset = dataset.map(partial(regression_to_classification_bin, bins=bins),
                              remove_columns='label')

        meta = get_classification_meta(dataset, name=dataset_name + "_sts")

        sts_datasets.append(SingleDataset(dataset, meta=meta))

        if DEBUG:
            break

    return sts_datasets

if __name__ == '__main__':
    sts_datasets = get_all_sts_datasets()
    config = {}
    out = Path("machamp/data/bigbio/sts")
    out.mkdir(exist_ok=True, parents=True)
    for dataset in tqdm(sts_datasets):
        config[dataset.meta.name] = {
            "train_data_path": str((out / dataset.meta.name).with_suffix(".train")),
            "validation_data_path": str((out / dataset.meta.name).with_suffix(".valid")),
            "sent_idxs": [0,1],
            "tasks": {
                dataset.meta.name: {
                    "column_idx": 2,
                    "task_type": "classification"
                }
            }
        }

        ### Generate validation split if not available
        if not "valid" in dataset.data:
            train_valid = dataset.data["train"].train_test_split(test_size=0.1)
            dataset.data = DatasetDict({
                "train": train_valid["train"],
                "valid": train_valid["test"],
            })

        def flatten_labels(example):
            example['labels'] = example['labels'][0]
            return example

        dataset.data = dataset.data.map(flatten_labels)
        dataset.data["train"].to_csv((out / dataset.meta.name).with_suffix(".train"), sep="\t", index=None, header=None)
        dataset.data["valid"].to_csv((out / dataset.meta.name).with_suffix(".valid"), sep="\t", index=None, header=None)


    ## Write Machamp config
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=1)

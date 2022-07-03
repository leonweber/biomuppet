import itertools
import json
import multiprocessing
from functools import partial
from pathlib import Path
from typing import List

import datasets
from bigbio.utils.constants import Tasks
from datasets import DatasetDict
from tqdm import tqdm

from biomuppet.classification import get_classification_meta
from biomuppet.relation_extraction import re_to_classification, is_valid_re
from biomuppet.utils import SingleDataset, get_all_dataloaders_for_task, split_sentences, \
    DEBUG, clean_text


def coref_to_re(example):
    example["relations"] = []
    for coref in example["coreferences"]:
        for i, (e1, e2) in enumerate(itertools.combinations(coref["entity_ids"], 2)):
            rel_id = coref["id"] + "_" + str(i)
            example["relations"].append(
                {
                    "id": rel_id,
                    "type": "coref",
                    "arg1_id": e1,
                    "arg2_id": e2,
                    "normalized": [],
                }
            )
    return example


def get_all_coref_datasets() -> List[SingleDataset]:
    coref_datasets = []

    for dataset_loader in tqdm(
            get_all_dataloaders_for_task(Tasks.COREFERENCE_RESOLUTION),
            desc="Preparing COREF datasets",
    ):
        dataset_name = Path(dataset_loader).with_suffix("").name

        if "lll" in dataset_name or "chemprot" in dataset_name or "pdr" in dataset_name or "2011_rel" in dataset_name or "2013_ge" in dataset_name or "cdr" in dataset_name:
            continue

        try:
            dataset = datasets.load_dataset(
                str(dataset_loader), name=f"{dataset_name}_bigbio_kb"
            )
        except ValueError as ve:
            print(f"Skipping {dataset_loader} because of {ve}")
            continue

        dataset = dataset.map(split_sentences, num_proc=multiprocessing.cpu_count())
        dataset = dataset.map(coref_to_re).map(
            partial(re_to_classification, mask_entities=False),
            batched=True,
            batch_size=1,
            remove_columns=dataset["train"].column_names,
            num_proc=multiprocessing.cpu_count()
        )

        dataset = dataset.filter(is_valid_re)
        meta = get_classification_meta(dataset=dataset, name=dataset_name + "_COREF")
        coref_datasets.append(SingleDataset(data=dataset, meta=meta))

        if DEBUG:
            break

    return coref_datasets


if __name__ == '__main__':
    coref_datasets = get_all_coref_datasets()
    config = {}
    out = Path("machamp/data/bigbio/coref")
    out.mkdir(exist_ok=True, parents=True)

    for dataset in tqdm(coref_datasets):
        config[dataset.meta.name] = {
            "train_data_path": str((out / dataset.meta.name).with_suffix(".train")),
            "validation_data_path": str((out / dataset.meta.name).with_suffix(".valid")),
            "sent_idxs": [0],
            "tasks": {
                dataset.meta.name: {
                    "column_idx": 1,
                    "task_type": "classification"
                }
            }
        }


        ### Generate validation split if not available
        if not "validation" in dataset.data:
            train_valid = dataset.data["train"].train_test_split(test_size=0.1)
            dataset.data = DatasetDict({
                "train": train_valid["train"],
                "validation": train_valid["test"],
            })

        ### Write train file
        with (out / dataset.meta.name).with_suffix(".train").open("w", encoding="utf8") as f:
            for example in dataset.data["train"]:
                text = clean_text(example["text"])
                if not text:
                    continue

                label = "|".join(sorted(example["labels"]))
                if not label.strip():
                    label = "None"

                f.write(text + "\t" + label + "\n")

        ### Write validation file
        with (out / dataset.meta.name).with_suffix(".valid").open("w", encoding="utf8") as f:
            for example in dataset.data["validation"]:
                text = example["text"].strip().replace("\t", " ").replace("\n", " ")
                if not text:
                    continue
                label = "|".join(sorted(example["labels"]))
                if not label.strip():
                    label = "None"

                f.write(text + "\t" + label + "\n")

        ### Write test file
        if "test" in dataset.data:
            with (out / dataset.meta.name).with_suffix(".test").open("w", encoding="utf8") as f:
                for example in dataset.data["test"]:
                    text = example["text"].strip().replace("\t", " ").replace("\n", " ")
                    if not text:
                        continue
                    label = "|".join(sorted(example["labels"]))
                    if not label.strip():
                        label = "None"

                    f.write(text + "\t" + label + "\n")
        
        
        

    ## Write Machamp config
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=1)

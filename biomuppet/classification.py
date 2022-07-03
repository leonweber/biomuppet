import json
import random
from pathlib import Path
from typing import List

import datasets
import scipy.stats
from bigbio.utils.constants import Tasks
from datasets import DatasetDict
from tqdm import tqdm

from biomuppet.utils import DatasetMetaInformation, SingleDataset, \
    get_all_dataloaders_for_task, DEBUG, clean_text


def get_classification_meta(dataset, name):
    is_multilabel = False
    label_to_idx = {"None": 0}
    all_train_labels = []
    for labels in dataset['train']['labels']:
        if len(labels) > 1:
            is_multilabel = True
        for label in labels:
            if label not in label_to_idx:
                label_to_idx[label] = len(label_to_idx)
            all_train_labels.append(label_to_idx[label])
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    task_type = "multilabel_clf" if is_multilabel else "clf"

    return DatasetMetaInformation(
        id_to_label=idx_to_label,
        label_to_id=label_to_idx,
        type=task_type,
        name=name,
        entropy=scipy.stats.entropy(all_train_labels)
    )


def subsample_negative(classification_dataset):
    positive_indices = [i for i, l in enumerate(classification_dataset["labels"]) if l]
    negative_indices = [
        i for i, l in enumerate(classification_dataset["labels"]) if not l
    ]
    if len(positive_indices) * 10 < len(negative_indices):
        negative_indices = random.sample(negative_indices, len(positive_indices) * 10)
    dataset = classification_dataset.select(positive_indices + negative_indices)

    return dataset

def get_all_classification_datasets() -> List[SingleDataset]:
    classification_datasets = []

    for dataset_loader in tqdm(
            get_all_dataloaders_for_task(Tasks.TEXT_CLASSIFICATION),
            desc="Preparing TEXT datasets",
    ):
        dataset_name = Path(dataset_loader).with_suffix("").name

        if (
                "cantemist" in dataset_name
                or "pharmaconer" in dataset_name
                or "hallmarks" in dataset_name
                or "nlmchem" in dataset_name
                or "bc7_litcovid" in dataset_name
        ):
            continue

        try:
            dataset = datasets.load_dataset(
                str(dataset_loader), name=f"{dataset_name}_bigbio_text"
            )
            meta = get_classification_meta(dataset=dataset, name=dataset_name + "_TEXT")
            classification_datasets.append(SingleDataset(data=dataset, meta=meta))

            if DEBUG and len(classification_datasets) >= 3:
                break

        except (ValueError, ImportError) as err:
            print(f"Skipping {dataset_loader} because of {err}")
            continue

    return classification_datasets

if __name__ == '__main__':
    classification_datasets = get_all_classification_datasets()
    config = {}
    out = Path("machamp/data/bigbio/classification")
    out.mkdir(exist_ok=True, parents=True)

    for dataset in tqdm(classification_datasets):
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

        # ### Write train file
        # with (out / dataset.meta.name).with_suffix(".train").open("w", encoding="utf8") as f:
        #     for example in dataset.data["train"]:
        #         text = clean_text(example["text"])
        #         if not text:
        #             continue

        #         label = "|".join(sorted(example["labels"]))
        #         if not label.strip():
        #             label = "None"

        #         f.write(text + "\t" + label + "\n")

        # ### Write validation file
        # with (out / dataset.meta.name).with_suffix(".valid").open("w", encoding="utf8") as f:
        #     for example in dataset.data["validation"]:
        #         text = example["text"].strip().replace("\t", " ").replace("\n", " ")
        #         if not text:
        #             continue
        #         label = "|".join(sorted(example["labels"]))
        #         if not label.strip():
        #             label = "None"

        #         f.write(text + "\t" + label + "\n")

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

    # ## Write Machamp config
    # with open(out / "config.json", "w") as f:
    #     json.dump(config, f, indent=1)

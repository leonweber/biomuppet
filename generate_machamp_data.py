import dataclasses
import itertools
import json
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import List
import importlib
from inspect import getmembers

import datasets
from datasets import DatasetDict
from flair.tokenization import SegtokSentenceSplitter
from tqdm import tqdm
import numpy as np

from utils.constants import Tasks

DEBUG = False


def overlaps(a, b):
    a = [int(i) for i in a]
    b = [int(i) for i in b]
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


@dataclasses.dataclass
class DatasetMetaInformation:
    label_to_id: dict
    id_to_label: dict
    name: str
    type: str

    def to(self, device):
        return self

def is_valid_re(example) -> bool:
    text = example["text"]
    return text.count("$") >= 2 and text.count("@") >= 2

def insert_consistently(offset, insertion, text, starts, ends):
    new_text = text[:offset] + insertion + text[offset:]
    new_starts = starts.copy()
    new_ends = ends.copy()
    new_starts[starts >= offset] += len(insertion)
    new_ends[ends >= offset] += len(insertion)

    return new_text, new_starts, new_ends


def delete_consistently(from_idx, to_idx, text, starts, ends):
    assert to_idx >= from_idx
    new_text = text[:from_idx] + text[to_idx:]
    new_starts = starts.copy()
    new_ends = ends.copy()
    new_starts[(from_idx <= starts) & (starts <= to_idx)] = from_idx
    new_ends[(from_idx <= ends) & (ends <= to_idx)] = from_idx
    new_starts[starts > to_idx] -= to_idx - from_idx
    new_ends[ends > to_idx] -= to_idx - from_idx

    return new_text, new_starts, new_ends


def insert_pair_markers(text, head, tail, passage_offset, mask_entities=True):
    head_start_marker = "@"
    tail_start_marker = "@"

    head_end_marker = "$"
    tail_end_marker = "$"

    head_start = min(i[0] for i in head["offsets"])
    head_end = max(i[1] for i in head["offsets"])

    tail_start = min(i[0] for i in tail["offsets"])
    tail_end = max(i[1] for i in tail["offsets"])

    starts = np.array([head_start, tail_start]) - passage_offset
    ends = np.array([head_end, tail_end]) - passage_offset

    if mask_entities:
        span_e1 = (starts[0], ends[0])
        span_e2 = (starts[1], ends[1])
        head_type = head["type"] or "entity"
        tail_type = tail["type"] or "entity"
        if overlaps(span_e1, span_e2):
            delete_span = (starts.min(), ends.max())
            text, starts, ends = delete_consistently(
                from_idx=delete_span[0],
                to_idx=delete_span[1],
                text=text,
                starts=starts,
                ends=ends,
            )
            text, starts, ends = insert_consistently(
                offset=starts[0],
                text=text,
                insertion=f"{head_type}-{tail_type}",
                starts=starts,
                ends=ends,
            )
            starts -= len(head_type)
            ends[:] = starts + len(tail_type)
        else:
            text, starts, ends = delete_consistently(
                from_idx=starts[0], to_idx=ends[0], text=text, starts=starts, ends=ends
            )
            text, starts, ends = insert_consistently(
                offset=starts[0],
                text=text,
                insertion=head_type,
                starts=starts,
                ends=ends,
            )
            starts[0] -= len(head_type)
            ends[0] = starts[0] + len(head_type)

            text, starts, ends = delete_consistently(
                from_idx=starts[1], to_idx=ends[1], text=text, starts=starts, ends=ends
            )
            text, starts, ends = insert_consistently(
                offset=starts[1],
                text=text,
                insertion=tail_type,
                starts=starts,
                ends=ends,
            )
            starts[1] -= len(tail_type)
            ends[1] = starts[1] + len(tail_type)

    text, starts, ends = insert_consistently(
        offset=starts[0],
        text=text,
        insertion=head_start_marker,
        starts=starts,
        ends=ends,
    )
    text, starts, ends = insert_consistently(
        offset=ends[0], text=text, insertion=head_end_marker, starts=starts, ends=ends
    )
    text, starts, ends = insert_consistently(
        offset=starts[1],
        text=text,
        insertion=tail_start_marker,
        starts=starts,
        ends=ends,
    )
    text, starts, ends = insert_consistently(
        offset=ends[1], text=text, insertion=tail_end_marker, starts=starts, ends=ends
    )

    return text


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


def re_to_classification(example, mask_entities=True):
    new_example = {"text": [""], "labels": [[""]]}
    relations = defaultdict(set)
    for relation in example["relations"][0]:
        relations[(relation["arg1_id"], relation["arg2_id"])].add(relation["type"])
        relations[(relation["arg2_id"], relation["arg1_id"])].add(relation["type"])
    for passage in example["passages"][0]:
        passage_range = range(*passage["offsets"][0])

        passage_entities = []
        for entity in example["entities"][0]:
            start = entity["offsets"][0][0]
            end = entity["offsets"][-1][1]
            if start in passage_range and (end - 1) in passage_range:
                passage_entities.append(entity)

        for i, head in enumerate(passage_entities):
            for tail in passage_entities[i:]:
                if head == tail:
                    continue
                text = insert_pair_markers(
                    passage["text"][0],
                    head=head,
                    tail=tail,
                    passage_offset=passage_range[0],
                    mask_entities=mask_entities
                )
                labels = relations[(head["id"], tail["id"])]
                new_example["text"].append(text)
                new_example["labels"].append(labels)

    return new_example


def split_sentences(example):
    new_passages = []

    splitter = SegtokSentenceSplitter()
    for passage in example["passages"]:
        for i, sentence in enumerate(splitter.split(passage["text"][0])):
            new_passages.append(
                {
                    "id": passage["id"] + ".s" + str(i),
                    "type": passage["type"],
                    "text": [sentence.to_original_text()],
                    "offsets": [
                        [
                            passage["offsets"][0][0] + sentence.start_pos,
                            passage["offsets"][0][0] + sentence.end_pos,
                            ]
                    ],
                }
            )

    example["passages"] = new_passages

    return example


def subsample_negative(classification_dataset):
    positive_indices = [i for i, l in enumerate(classification_dataset["labels"]) if l]
    negative_indices = [
        i for i, l in enumerate(classification_dataset["labels"]) if not l
    ]
    if len(positive_indices) * 10 < len(negative_indices):
        negative_indices = random.sample(negative_indices, len(positive_indices) * 10)
    dataset = classification_dataset.select(positive_indices + negative_indices)

    return dataset


class SingleDataset:
    def __init__(self, data, name):
        self.data = data
        self.name = name



def get_all_dataloaders_for_task(task: Tasks) -> List[datasets.Dataset]:
    dataset_loaders_for_task = []

    all_dataset_loaders = list(Path("biodatasets").glob("*/*py"))
    for dataset_loader in all_dataset_loaders:
        try:
            module = str(dataset_loader).replace("/", ".").replace(".py", "")
            if task in importlib.import_module(module)._SUPPORTED_TASKS:
                dataset_loaders_for_task.append(str(dataset_loader))
        except ImportError as err:
            print(f"Skipping {dataset_loader} because of {err}")

    return dataset_loaders_for_task

def get_all_re_datasets() -> List[SingleDataset]:
    re_datasets = []

    for dataset_loader in tqdm(
            get_all_dataloaders_for_task(Tasks.RELATION_EXTRACTION),
            desc="Preparing RE datasets",
    ):
        dataset_name = Path(dataset_loader).with_suffix("").name

        if (
                "lll" in dataset_name
                or "chemprot" in dataset_name
                or "pdr" in dataset_name
                or "2011_rel" in dataset_name
                or "2013_ge" in dataset_name
                or "cdr" in dataset_name
        ):
            continue

        try:
            dataset = datasets.load_dataset(
                str(dataset_loader), name=f"{dataset_name}_bigbio_kb"
            )
        except ValueError as ve:
            print(f"Skipping {dataset_loader} because of {ve}")
            continue

        dataset = dataset.map(split_sentences)
        dataset = dataset.map(
            re_to_classification,
            batched=True,
            batch_size=1,
            remove_columns=dataset["train"].column_names,
        )
        dataset = dataset.filter(is_valid_re)

        for split_name, split in dataset.items():
            dataset[split_name] = subsample_negative(split)

        re_datasets.append(SingleDataset(data=dataset, name=dataset_name + "_RE"))

        if DEBUG:
            break

    return re_datasets


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
        ):
            continue

        try:
            dataset = datasets.load_dataset(
                str(dataset_loader), name=f"{dataset_name}_bigbio_text"
            )
            classification_datasets.append(SingleDataset(data=dataset, name=dataset_name + "_classification"))

            if DEBUG:
                break

        except (ValueError, ImportError) as err:
            print(f"Skipping {dataset_loader} because of {err}")
            continue

    return classification_datasets


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

        dataset = dataset.map(split_sentences)
        dataset = dataset.map(coref_to_re).map(
            partial(re_to_classification, mask_entities=False),
            batched=True,
            batch_size=1,
            remove_columns=dataset["train"].column_names,
        )

        dataset = dataset.filter(is_valid_re)
        for split_name, split in dataset.items():
            dataset[split_name] = subsample_negative(split)

        coref_datasets.append(SingleDataset(dataset, name=dataset_name + "_coref"))

        if DEBUG:
            break

    return coref_datasets


if __name__ == "__main__":
    re_datasets = get_all_re_datasets()
    coref_datasets = get_all_coref_datasets()
    classification_datasets = get_all_classification_datasets()

    config = {}

    out = Path("machamp/data/bigbio/")
    out.mkdir(exist_ok=True, parents=True)

    # Process datasets

    ## Process classification-type datasets
    ### Add to Machamp config
    for dataset in tqdm(re_datasets + coref_datasets + classification_datasets):
        config[dataset.name] = {
            "train_data_path": str((out / dataset.name).with_suffix(".train")),
            "validation_data_path": str((out / dataset.name).with_suffix(".valid")),
            "sent_idxs": [0],
            "tasks": {
                dataset.name: {
                    "column_idx": 1,
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

        ### Write train file
        with (out / dataset.name).with_suffix(".train").open("w") as f:
            for example in dataset.data["train"]:
                text = example["text"].strip().replace("\t", " ").replace("\n", " ")
                if not text:
                    continue
                label = "|".join(sorted(example["labels"]))
                if not label.strip():
                    label = "None"

                f.write(text + "\t" + label + "\n")

        ### Write validation file
        with (out / dataset.name).with_suffix(".valid").open("w") as f:
            for example in dataset.data["valid"]:
                text = example["text"].strip().replace("\t", " ").replace("\n", " ")
                if not text:
                    continue
                label = "|".join(sorted(example["labels"]))
                if not label.strip():
                    label = "None"

                f.write(text + "\t" + label + "\n")


    ## Write Machamp config
    fname = "machamp/configs/bigbio_debug.json" if DEBUG else "machamp/configs/bigbio_full.json"
    with open(fname, "w") as f:
        json.dump(config, f, indent=1)
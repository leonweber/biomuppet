import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import datasets
import numpy as np
from bigbio.utils.constants import Tasks
from tqdm import tqdm
import itertools

from biomuppet.classification import get_classification_meta
from biomuppet.named_entity_recognition import (
    get_sequence_labelling_meta,
    bigbio_ner_to_conll,
)
from biomuppet.relation_extraction import is_valid_re
from biomuppet.utils import (
    overlaps,
    SingleDataset,
    get_all_dataloaders_for_task,
    split_sentences,
)

FAILING_QA = ["pcr"]


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


def events_to_relations(events) -> Dict:

    relations = defaultdict(set)

    for event in events:

        head = event["id"]

        for argument in event["arguments"]:

            tail = argument["ref_id"]
            relation_type = argument["role"]
            relations[(head, tail)].add(relation_type)

    return dict(relations)


def get_passage_event_entities(passage_range, relations, entities, events):

    valid_ids = set(e for head_tail, types in relations.items() for e in head_tail)

    passage_entities = []

    for entity in entities:
        if entity["id"] in valid_ids:
            start = entity["offsets"][0][0]
            end = entity["offsets"][-1][1]
            if start in passage_range and (end - 1) in passage_range:
                entity["event"] = False
                passage_entities.append(entity)

    for event in events:
        if event["id"] in valid_ids:
            start = event["trigger"]["offsets"][0][0]
            end = event["trigger"]["offsets"][-1][1]
            if start in passage_range and (end - 1) in passage_range:
                entity = {
                    "id": event["id"],
                    "text": event["trigger"]["text"],
                    "offsets": event["trigger"]["offsets"],
                    "type": event["type"],
                    "event": True,
                }
                passage_entities.append(entity)

    return passage_entities


def event_extraction_to_relation_classification(example, mask_entities=True):

    new_example = {"text": [""], "labels": [[""]]}

    relations = events_to_relations(events=example["events"][0])

    for passage in example["passages"][0]:

        passage_range = range(*passage["offsets"][0])

        passage_entities = get_passage_event_entities(
            passage_range=passage_range,
            relations=relations,
            entities=example["entities"][0],
            events=example["events"][0],
        )

        for (head, tail) in itertools.permutations(passage_entities, 2):

            if not head["event"]:
                continue

            text = insert_pair_markers(
                passage["text"][0],
                head=head,
                tail=tail,
                passage_offset=passage_range[0],
                mask_entities=mask_entities,
            )

            key = (head["id"], tail["id"])
            labels = relations[key] if key in relations else set()

            new_example["text"].append(text)
            new_example["labels"].append(labels)

    return new_example


def get_all_ee_as_re_datasets() -> List[SingleDataset]:

    re_datasets = []

    dataset_loaders = get_all_dataloaders_for_task(Tasks.EVENT_EXTRACTION)

    for dataset_loader in tqdm(
        dataset_loaders, desc="Preparing EventExtraction::EdgeClassification datasets"
    ):
        dataset_name = Path(dataset_loader).with_suffix("").name

        if dataset_name in FAILING_QA:
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
            event_extraction_to_relation_classification,
            batched=True,
            batch_size=1,
            remove_columns=dataset["train"].column_names,
        )

        dataset = dataset.filter(is_valid_re)

        meta = get_classification_meta(dataset=dataset, name=dataset_name + "_RE")
        re_datasets.append(SingleDataset(data=dataset, meta=meta))

    return re_datasets


def event_triggers_to_entities(example):

    example["entities"] = []

    for event in example["events"]:
        entity = {
            "id": event["id"],
            "offsets": event["trigger"]["offsets"],
            "text": event["trigger"]["text"],
            "type": event["type"],
            "normalized": [],
        }
        example["entities"].append(entity)

    return example


def get_all_ee_as_ner_datasets() -> List[SingleDataset]:

    re_datasets = []

    dataset_loaders = get_all_dataloaders_for_task(Tasks.EVENT_EXTRACTION)

    for dataset_loader in tqdm(
        dataset_loaders, desc="Preparing EventExtraction::TriggerRecognition datasets"
    ):
        dataset_name = Path(dataset_loader).with_suffix("").name

        if dataset_name in FAILING_QA:
            continue

        try:
            dataset = datasets.load_dataset(
                str(dataset_loader), name=f"{dataset_name}_bigbio_kb"
            )
        except ValueError as ve:
            print(f"Skipping {dataset_loader} because of {ve}")
            continue

        dataset = dataset.map(split_sentences)
        dataset = dataset.map(event_triggers_to_entities)
        dataset = dataset.map(
            bigbio_ner_to_conll,
            remove_columns=[
                "passages",
                "entities",
                "events",
                "coreferences",
                "relations",
            ],
        )

        meta = get_sequence_labelling_meta(dataset=dataset, name=dataset_name + "_NER")
        re_datasets.append(SingleDataset(data=dataset, meta=meta))

    return re_datasets


def write_as_relation_extraction_datasets():

    ee_as_re_datasets = get_all_ee_as_re_datasets()

    config = {}

    out = Path("machamp/data/bigbio/event_extraction_edge_classification")
    out.mkdir(exist_ok=True, parents=True)

    for dataset in tqdm(ee_as_re_datasets):
        config[dataset.name] = {
            "train_data_path": str((out / dataset.name).with_suffix(".train")),
            "validation_data_path": str((out / dataset.name).with_suffix(".valid")),
            "sent_idxs": [0],
            "tasks": {dataset.name: {"column_idx": 1, "task_type": "classification"}},
        }

        if "validation" not in dataset.data:
            train_valid = dataset.data["train"].train_test_split(test_size=0.1)
            dataset.data = DatasetDict(
                {
                    "train": train_valid["train"],
                    "validation": train_valid["test"],
                }
            )

        with (out / dataset.name).with_suffix(".train").open("w", encoding="utf8") as f:
            for example in dataset.data["train"]:
                text = example["text"].strip().replace("\t", " ").replace("\n", " ")
                if not text:
                    continue
                label = "|".join(sorted(example["labels"]))
                if not label.strip():
                    label = "None"

                f.write(text + "\t" + label + "\n")

        with (out / dataset.name).with_suffix(".valid").open("w", encoding="utf8") as f:
            for example in dataset.data["validation"]:
                text = example["text"].strip().replace("\t", " ").replace("\n", " ")
                if not text:
                    continue
                label = "|".join(sorted(example["labels"]))
                if not label.strip():
                    label = "None"

                f.write(text + "\t" + label + "\n")

        if "test" in dataset.data:
            with (out / dataset.name).with_suffix(".test").open("w", encoding="utf8") as f:
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


def write_as_ner_datasets():

    ee_as_ner_datasets = get_all_ee_as_ner_datasets()

    config = {}

    out = Path("machamp/data/bigbio/event_extraction_trigger_recognition")
    out.mkdir(exist_ok=True, parents=True)

    for dataset in tqdm(ee_as_ner_datasets):

        config[dataset.meta.name] = {
            "train_data_path": str((out / dataset.meta.name).with_suffix(".train")),
            "validation_data_path": str(
                (out / dataset.meta.name).with_suffix(".valid")
            ),
            "word_idx": 0,
            "tasks": {dataset.meta.name: {"column_idx": 1, "task_type": "seq"}},
        }

        if "validation" not in dataset.data:
            train_valid = dataset.data["train"].train_test_split(test_size=0.1, seed=0)
            dataset.data = DatasetDict(
                {
                    "train": train_valid["train"],
                    "validation": train_valid["test"],
                }
            )

        # with (out / dataset.meta.name).with_suffix(".train").open("w") as f:
        #     for example in dataset.data["train"]:
        #         for word, label in example["conll"]:
        #             if word or label:
        #                 f.write(word + "\t" + label + "\n")
        #             else:
        #                 f.write("\n")
        #         f.write("\n")

        # with (out / dataset.meta.name).with_suffix(".valid").open("w") as f:
        #     for example in dataset.data["validation"]:
        #         for word, label in example["conll"]:
        #             if word or label:
        #                 f.write(word + "\t" + label + "\n")
        #             else:
        #                 f.write("\n")
        #         f.write("\n")

        if "test" in dataset.data:
            with (out / dataset.meta.name).with_suffix(".test").open("w") as f:
                for example in dataset.data["test"]:
                    for word, label in example["conll"]:
                        if word or label:
                            f.write(word + "\t" + label + "\n")
                        else:
                            f.write("\n")
                    f.write("\n")

    ## Write Machamp config
    # with open(out / "config.json", "w") as f:
    #     json.dump(config, f, indent=1)

def test_trigger_detection():

    import pandas as pd

    dataset_loaders = get_all_dataloaders_for_task(Tasks.EVENT_EXTRACTION)

    for dataset_loader in tqdm(
        dataset_loaders, desc="Preparing EventExtraction::EdgeClassification datasets"
    ):
        dataset_name = Path(dataset_loader).with_suffix("").name

        if dataset_name in FAILING_QA:
            continue

        try:
            dataset = datasets.load_dataset(
                str(dataset_loader), name=f"{dataset_name}_bigbio_kb"
            )
        except ValueError as ve:
            print(f"Skipping {dataset_loader} because of {ve}")
            continue

        for split in ["train", "valid"]:

            lines = open(
                f"machamp/data/bigbio/event_extraction_trigger_recognition/{dataset_name}_NER.{split}"
            ).readlines()

            machamp = pd.DataFrame(
                [
                    dict(zip(["token", "label"], line.strip().split("\t")))
                    for line in lines
                    if line != "\n"
                ]
            )

            bigbio = dataset[split]

            triggers = [
                e["trigger"]["text"][0].split()
                for example in bigbio
                for e in example["events"]
            ]
            triggers = [w for words in triggers for w in words]
            tokens = [
                token
                for token, label in zip(list(machamp["token"]), list(machamp["label"]))
                if label != "O"
            ]

            raise NotImplementedError("Not really sure what a good test might be here")


if __name__ == "__main__":

    from datasets import DatasetDict

    write_as_relation_extraction_datasets()

    write_as_ner_datasets()


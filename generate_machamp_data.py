import dataclasses
import itertools
import json
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import List
from nltk.tokenize import WordPunctTokenizer
import importlib.util
from inspect import getmembers

import datasets
from datasets import DatasetDict
from flair.tokenization import SegtokSentenceSplitter, SpacySentenceSplitter
from tqdm import tqdm
import numpy as np

import bigbio
from bigbio.utils.constants import Tasks

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

def is_valid_qa(example) -> bool:
    """
    Return non-empty text
    :param example:
    :return:
    """
    if "text" in example.keys():
        text = example["text"]
        return len(text) >=1
    else:
        sequence = example["sequence"]
        return len(sequence) >=1


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

def is_seq_answer(seq, answer):
    """
    Returns boolean if there's a match between seq and answer
    :param seq:
    :param answer:
    :return:
    """
    if seq.strip().lower() == answer.strip().lower():
        return True


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

#TODO: Transform QA to classification
def qa_to_classification(example):
    """
    Function that converts yesno and multiple choice QA to MaChamp classification
    :param example:
    :param mask_entities:
    :return:
    """
    if example["type"][0] == "yesno":
        new_example = {"text": [""], "labels": [[""]]}
        context = example["context"][0].strip().replace("\t", " ").replace("\n", " ")
        question = example["question"][0].strip().replace("\t", " ").replace("\n", " ")
        composite_text = context + " " + "[SEP]" + " " + question
        labels = [example["answer"][0][0].strip().replace("\t", " ").replace("\n", " ")]
        new_example["text"].append(composite_text)
        new_example["labels"].append(labels)
        return new_example

    elif example["type"][0] == "multiple_choice":
        new_example = {"text": [""], "labels": [[""]]}
        context = example["context"][0].strip().replace("\t", " ").replace("\n", " ")
        question = example["question"][0].strip().replace("\t", " ").replace("\n", " ")
        answer = example["answer"][0][0].strip().replace("\t", " ").replace("\n", " ")
        for choice in example["choices"][0]:
            answer_prompt = "Answer : {}".format(choice)
            composite_text = context + " " + "[SEP]" + " " + question + " " + answer_prompt
            if answer.lower() == choice.lower():
                labels = ["yes"]
            else:
                labels = ["no"]
            new_example["text"].append(composite_text)
            new_example["labels"].append(labels)
        return new_example

    else:
        return



#TODO: transforming QA to multiple choice
def qa_to_sequence(example):
    """
    Function that converts MCQA to MaChamp sequence
    Currently only converts data that has answer exactly matches that in the prompt

    1. Assert type =="multiple choice"
    2. Format sequence: tokens of the words (separated by empty line between samples)
    3. Format labels: [] or [answer]

    """
    if example["type"][0] != "multiple_choice":
        return
    else:
        new_example = {"sequence": [[""]], "labels": [[""]]}
        tokenizer = WordPunctTokenizer()
        context_seq = tokenizer.tokenize(example["context"][0].lower().strip().replace("\t", " ").replace("\n", " "))
        question_seq = tokenizer.tokenize(example["question"][0].lower().strip().replace("\t", " ").replace("\n", " "))
        sequence = context_seq + question_seq

        # Look for exact matches firstly:
        ans =example['answer'][0][0].lower().strip().replace("\t", " ").replace("\n", " ")
        if ans in sequence:
            labels = [["answer"] if (ans == seq) else [""] for seq in sequence]
            new_example["sequence"].extend([[seq] for seq in sequence])
            new_example["labels"].extend(labels)
            # Add new line to differentiate examples
            new_example["sequence"].extend([["\n"]])
            new_example["labels"].extend([["\n"]])
            return new_example
        # Currently doesn't deal with "soft matching" cases
        # TODO: implement when multiple choice answers "soft match" in the form used in the prompt
        else:
            return new_example


def get_classification_meta(dataset, name):
    is_multilabel = False
    label_to_idx = {"None": 0}
    for labels in dataset['train']['labels']:
        if len(labels) > 1:
            is_multilabel = True
        for label in labels:
            if label not in label_to_idx:
                label_to_idx[label] = len(label_to_idx)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    task_type = "multilabel_clf" if is_multilabel else "clf"

    return DatasetMetaInformation(
        id_to_label=idx_to_label,
        label_to_id=label_to_idx,
        type=task_type,
        name=name
    )


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
    def __init__(self, data, meta, split="train"):
        self.split = split
        self.data = data
        self.meta = meta

    def __getitem__(self, item):
        example = self.data[self.split][item]
        example["meta"] = self.meta
        return example

    def __len__(self):
        return len(self.data[self.split])



def get_all_dataloaders_for_task(task: Tasks) -> List[datasets.Dataset]:
    dataset_loaders_for_task = []
    biodatasets_path = Path(bigbio.__file__).resolve().parents[1] / "biodatasets"
    all_dataset_loaders = list(biodatasets_path.glob("*/*py"))
    for dataset_loader in all_dataset_loaders:
        spec = importlib.util.spec_from_file_location("foo", dataset_loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if task in module._SUPPORTED_TASKS:
            dataset_loaders_for_task.append(str(dataset_loader))

    return dataset_loaders_for_task


def get_all_re_datasets() -> List[SingleDataset]:
    re_datasets = []

    dataset_loaders = get_all_dataloaders_for_task(Tasks.RELATION_EXTRACTION)
    for dataset_loader in tqdm(dataset_loaders, desc="Preparing RE datasets"):
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

        # for split_name, split in dataset.items():
        #     dataset[split_name] = subsample_negative(split)

        meta = get_classification_meta(dataset=dataset, name=dataset_name + "_RE")
        re_datasets.append(SingleDataset(data=dataset, meta=meta))

        if DEBUG and len(re_datasets) >= 3:
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
                or "nlmchem" in dataset_name
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
        # for split_name, split in dataset.items():
        #     dataset[split_name] = subsample_negative(split)
        meta = get_classification_meta(dataset=dataset, name=dataset_name + "_COREF")
        coref_datasets.append(SingleDataset(data=dataset, meta=meta))


        if DEBUG:
            break

    return coref_datasets


def get_all_qa_datasets() -> List[SingleDataset]:
    """
    Function that transforms QA dataset to MaChamp format;
    Transforms ALL QA dataset with yesno and multiple choice --> MaChamp classification
    Transforms some QA datasest with multiple choice that has exact match between answer and the prompt --> MaChamp
    Sequence
    :return: qa_datasets
    """
    qa_clf_datasets = []
    qa_seq_datasets = []

    for dataset_loader in tqdm(
            get_all_dataloaders_for_task(Tasks.QUESTION_ANSWERING),
            desc="Preparing QA datasets",
    ):
        dataset_name = Path(dataset_loader).with_suffix("").name

        if (
                ("mediqa_qa" == dataset_name)  # empty context
                or ("biology_how_why_corpus" == dataset_name)  # empty context
                or ("med_qa" == dataset_name)  # non-English
                or ("bioasq_task_b" == dataset_name)  # local dataset
        ):
            continue

        module = datasets.load.dataset_module_factory(str(dataset_loader))
        builder_cls = datasets.load.import_main_class(module.module_path)
        all_bigbio_config_names = [el.name for el in builder_cls.BUILDER_CONFIGS if
                            'bigbio_qa' in el.name and 'unlabel' not in el.name]

        for bigbio_config_name in all_bigbio_config_names:
            try:
                dataset = datasets.load_dataset(
                    str(dataset_loader), name=bigbio_config_name
                )
            except ValueError as ve:
                print(f"Skipping {dataset_loader} because of {ve}")
                continue

            # convert all QA dataset to MaChamp classification and add as SingleDataset
            new_dataset = dataset.map(
                qa_to_classification,
                batched=True,
                batch_size=1,
                remove_columns=dataset["train"].column_names,
            )
            new_dataset = new_dataset.filter(is_valid_qa)
            for split_name, split in new_dataset.items():
                new_dataset[split_name] = subsample_negative(split)
            #TODO: write solution for meta information
            meta = get_classification_meta(dataset=new_dataset, name=bigbio_config_name +"_CLF")
            qa_clf_datasets.append(SingleDataset(new_dataset, meta=meta))

            # If the dataset is MCQA, create MaChamp sequence format
            # convert all QA dataset to MaChamp classification and add as SingleDataset
            mcqa_dataset = dataset.filter(lambda x: x["type"]=="multiple_choice")
            if mcqa_dataset.num_rows['train'] >0:
                new_dataset = mcqa_dataset.map(
                    qa_to_sequence,
                    batched=True,
                    batch_size=1,
                    remove_columns=mcqa_dataset["train"].column_names,
                )
                new_dataset = new_dataset.filter(is_valid_qa)
                # for split_name, split in new_dataset.items():
                #     new_dataset[split_name] = subsample_negative(split)
                meta = get_classification_meta(dataset=new_dataset, name=bigbio_config_name+"_SEQ")
                qa_seq_datasets.append(SingleDataset(new_dataset, meta=meta))

        if DEBUG:
            break

    return qa_clf_datasets, qa_seq_datasets

if __name__ == "__main__":
    re_datasets = get_all_re_datasets()
    coref_datasets = get_all_coref_datasets()
    classification_datasets = get_all_classification_datasets()
    qa_clf_datasets, qa_seq_datasets = get_all_qa_datasets()

    config = {}

    out = Path("machamp/data/bigbio/")
    out.mkdir(exist_ok=True, parents=True)

    # Process datasets

    ## Process classification-type datasets
    ### Add to Machamp config
    for dataset in tqdm(re_datasets + coref_datasets + classification_datasets + qa_clf_datasets):
        name = dataset.meta.name
        config[name] = {
            "train_data_path": str((out / name).with_suffix(".train")),
            "validation_data_path": str((out / name).with_suffix(".valid")),
            "sent_idxs": [0],
            "tasks": {
                name: {
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
        with (out / name).with_suffix(".train").open("w") as f:
            for example in dataset.data["train"]:
                text = example["text"].strip().replace("\t", " ").replace("\n", " ") if "text" in example.keys() \
                    else example["sequence"].strip().replace("\t", " ").replace("\n", " ")
                if not text:
                    continue
                label = "|".join(sorted(example["labels"]))
                if not label.strip():
                    label = "None"

                f.write(text + "\t" + label + "\n")

        ### Write validation file
        with (out / name).with_suffix(".valid").open("w") as f:
            for example in dataset.data["valid"]:
                text = example["text"].strip().replace("\t", " ").replace("\n", " ") if "text" in example.keys() \
                    else example["sequence"].strip().replace("\t", " ").replace("\n", " ")
                if not text:
                    continue
                label = "|".join(sorted(example["labels"]))
                if not label.strip():
                    label = "None"

                f.write(text + "\t" + label + "\n")

    for dataset in tqdm(qa_seq_datasets):
        name = dataset.meta.name
        config[name] = {
            "train_data_path": str((out / name).with_suffix(".train")),
            "validation_data_path": str((out / name).with_suffix(".valid")),
            "word_idxs": 0,
            "tasks": {
                name: {
                    "column_idx": 1,
                    "task_type": "sequence"
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
        with (out / name).with_suffix(".train").open("w") as f:
            for example in dataset.data["train"]:
                text = example["text"].strip().replace("\t", " ").replace("\n", " ") if "text" in example.keys() \
                    else example["sequence"][0].strip().replace("\t", " ").replace("\n", " ")
                if not text:
                    continue
                label = "|".join(sorted(example["labels"]))
                if not label.strip():
                    label = "None"

                f.write(text + "\t" + label + "\n")

        ### Write validation file
        with (out / name).with_suffix(".valid").open("w") as f:
            for example in dataset.data["valid"]:
                text = example["text"].strip().replace("\t", " ").replace("\n", " ") if "text" in example.keys() \
                    else example["sequence"][0].strip().replace("\t", " ").replace("\n", " ")
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
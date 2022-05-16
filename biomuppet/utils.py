import dataclasses
import importlib
from pathlib import Path
from typing import List

import bigbio
import datasets
from bigbio.utils.constants import Tasks
from bigbio.dataloader import BigBioConfigHelpers, BigBioConfigHelper
from flair.tokenization import SegtokSentenceSplitter

DEBUG = True


@dataclasses.dataclass
class DatasetMetaInformation:
    label_to_id: dict
    id_to_label: dict
    name: str
    type: str
    entropy: float

    def to(self, device):
        return self


class SingleDataset:
    def __init__(self, data, meta: DatasetMetaInformation, split="train"):
        self.split = split
        self.data = data
        self.meta = meta

    @property
    def name(self):
        return self.meta.name

    def __getitem__(self, item):
        example = self.data[self.split][item]
        example["meta"] = self.meta
        return example

    def __len__(self):
        return len(self.data[self.split])

def overlaps(a, b):
    a = [int(i) for i in a]
    b = [int(i) for i in b]
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def clean_text(text):
    return text.strip().replace("\t", " ").replace("\n", " ")

def split_sentences(example):
    new_passages = []

    splitter = SegtokSentenceSplitter()
    for passage in example["passages"]:
        passage_text = clean_text(passage["text"][0])
        for i, sentence in enumerate(splitter.split(passage_text)):
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


def get_all_dataloaders_for_task(task: Tasks) -> List[BigBioConfigHelper]:
    conhelps = BigBioConfigHelpers()
    bb_task_public_helpers = conhelps.filtered(
        lambda x: (
            x.is_bigbio_schema
            and task in x.tasks
            and not x.is_local
        )
    )

    return sorted(set(i.get_load_dataset_kwargs()["path"] for i in bb_task_public_helpers))


def clean_text(text):
    return text.strip().replace("\t", " ").replace("\n", " ")

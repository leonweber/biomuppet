import dataclasses
import importlib
from pathlib import Path
from typing import List

import bigbio
import datasets
from bigbio.utils.constants import Tasks
from flair.tokenization import SegtokSentenceSplitter

DEBUG = False


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

def overlaps(a, b):
    a = [int(i) for i in a]
    b = [int(i) for i in b]
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))




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


def clean_text(text):
    return text.strip().replace("\t", " ").replace("\n", " ")

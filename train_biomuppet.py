import argparse
import dataclasses
import itertools
import sys
from dataclasses import dataclass
import random
from collections import defaultdict
from pathlib import Path
import importlib

import numpy as np
import datasets
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

import torch
import torchmetrics
import transformers
from torch.utils.data import Dataset, BatchSampler, DataLoader, DistributedSampler
import pytorch_lightning as pl
from torch import nn
from transformers.models.auto.tokenization_auto import AutoTokenizer

from bigbio.utils.constants import Tasks


pl.seed_everything(42)


def classification_loss(logits, labels, meta):
    # TODO scale loss so that losses for diferent tasks are all on the same scale
    assert logits.ndim == 2
    cls_logit = logits[0]
    label_tensor = torch.zeros(len(meta.label_to_id))
    if labels:
        label = labels[0]
    else:
        label = "None"

    label_tensor[meta.label_to_id[label]] = 1

    return nn.CrossEntropyLoss()(cls_logit.unsqueeze(0), label_tensor.to(logits.device).unsqueeze(0))


def multilabel_classification_loss(logits, labels, meta):
    assert logits.ndim == 2
    cls_logit = logits[0]
    label_tensor = torch.zeros(len(meta.label_to_id))
    for label in labels:
        label_tensor[meta.label_to_id[label]] = 1

    return nn.BCEWithLogitsLoss()(cls_logit.unsqueeze(0), label_tensor.to(logits.device).unsqueeze(0))


TASK_TYPE_TO_LOSS = {"clf": classification_loss,
                     "multilabel_clf": multilabel_classification_loss}

@dataclasses.dataclass
class ToableList:
    labels: list

    def to(self, device):
        return self

    def __getitem__(self, item):
        return self.labels[item]

    def __len__(self):
        return len(self.labels)

@dataclasses.dataclass
class DatasetMetaInformation:
    label_to_id: dict
    id_to_label: dict
    name: str
    type: str

    def to(self, device):
        return self

def insert_consistently(offset, insertion, text, starts, ends):
    new_text = text[:offset] + insertion + text[offset:]
    new_starts = starts.copy()
    new_ends = ends.copy()
    new_starts[starts >= offset] += len(insertion)
    new_ends[ends >= offset] += len(insertion)

    return new_text, new_starts, new_ends


def delete_consistently(from_idx, to_idx , text, starts, ends):
    assert to_idx >= from_idx
    new_text = text[:from_idx] + text[to_idx:]
    new_starts = starts.copy()
    new_ends = ends.copy()
    new_starts[(from_idx <= starts) & (starts <= to_idx)] = from_idx
    new_ends[(from_idx <= ends) & (ends <= to_idx)] = from_idx
    new_starts[starts > to_idx] -= (to_idx - from_idx)
    new_ends[ends > to_idx] -= (to_idx - from_idx)

    return new_text, new_starts, new_ends


def insert_pair_markers(text, head, tail, passage_offset):
    head_start_marker = "[HEAD-S]"
    tail_start_marker = "[TAIL-S]"
    head_end_marker = "[HEAD-E]"
    tail_end_marker = "[TAIL-E]"

    starts = (
            np.array([head["offsets"][0][0], tail["offsets"][0][0]]) - passage_offset
    )
    ends = (
            np.array([head["offsets"][-1][-1], tail["offsets"][-1][-1]]) - passage_offset
    )

    text, starts, ends = insert_consistently(
        offset=starts[0], text=text, insertion=head_start_marker, starts=starts, ends=ends
    )
    text, starts, ends = insert_consistently(
        offset=ends[0], text=text, insertion=head_end_marker, starts=starts, ends=ends
    )
    text, starts, ends = insert_consistently(
        offset=starts[1], text=text, insertion=tail_start_marker, starts=starts, ends=ends
    )
    text, starts, ends = insert_consistently(
        offset=ends[1], text=text, insertion=tail_end_marker, starts=starts, ends=ends
    )

    return text

def re_to_classification(example):
    new_example = {"nested_texts": [], "nested_labels": []}
    relations = defaultdict(set)
    for relation in example["relations"]:
        relations[(relation["arg1_id"], relation["arg2_id"])].add(relation["type"])
    for passage in example["passages"]:
        passage_range = range(*passage["offsets"][0])

        passage_entities = []
        for entity in example["entities"]:
            start = entity["offsets"][0][0]
            end = entity["offsets"][-1][1]
            if start in passage_range and (end - 1) in passage_range:
                passage_entities.append(entity)

        for head in passage_entities:
            for tail in passage_entities:
                text = insert_pair_markers(passage["text"][0], head=head, tail=tail,
                                           passage_offset=passage_range[0])
                labels = relations[(head["id"], tail["id"])]
                labels = labels | set(rel + "_r" for rel in relations[(tail["id"], head["id"])])
                new_example["nested_texts"].append(text)
                new_example["nested_labels"].append(labels)

    return new_example

def split_sentences(example):
    new_passages = []

    splitter = SegtokSentenceSplitter()
    for passage in example["passages"]:
        for i, sentence in enumerate(splitter.split(passage["text"][0])):
            new_passages.append({
                "id": passage["id"] + ".s" + str(i),
                "type": passage["type"],
                "text": [sentence.to_original_text()],
                "offsets": [[passage["offsets"][0][0] + sentence.start_pos, passage["offsets"][0][0] + sentence.end_pos]]
            })

    example["passages"] = new_passages

    return example


@dataclass
class MultitaskClassifierOutput(transformers.file_utils.ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    dataset_to_logits: Dict[str, torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SingleDataset:
    def __init__(self, data, meta):
        self.data = data
        self.meta = meta

    def __getitem__(self, item):
        example = self.data[item]
        example["meta"] = self.meta
        return example

    def __len__(self):
        return len(self.data)


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


def get_classification_meta(dataset, name):
    is_multilabel = False
    label_to_idx = {"None": 0}
    for labels in dataset['labels']:
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

###
# Relation Extraction Tasks Data Loading
###
def get_all_re_datasets(tokenizer, split="train") -> List[SingleDataset]:
    re_datasets = []

    for dataset_loader in tqdm(get_all_dataloaders_for_task(Tasks.RELATION_EXTRACTION), desc="Preparing RE datasets"):
        dataset_name = Path(dataset_loader).with_suffix("").name

        if "lll" in dataset_name or "chemprot" in dataset_name or "pdr" in dataset_name or "2011_rel" in dataset_name or "2013_ge" in dataset_name:
            continue

        try:
            dataset = datasets.load_dataset(str(dataset_loader), name=f"{dataset_name}_bigbio_kb", split=split)
        except ValueError as ve:
            print(f"Skipping {dataset_loader} because of {ve}")
            continue

        dataset = dataset.map(re_to_classification)
        classification_dataset = {"text": [], "labels": []}
        for example in dataset:
            for text, labels in zip(example["nested_texts"], example["nested_labels"]):
                classification_dataset["text"].append(text)
                classification_dataset["labels"].append(labels)
        positive_indices = [i for i, l in enumerate(classification_dataset["labels"]) if l]
        negative_indices = [i for i, l in enumerate(classification_dataset["labels"]) if not l]
        if len(positive_indices) * 10 < len(negative_indices):
            negative_indices = random.sample(negative_indices, len(positive_indices) * 10)
        dataset = datasets.Dataset.from_dict(classification_dataset).select(positive_indices + negative_indices)
        dataset = dataset.map(lambda x: tokenizer(x["text"], max_length=512, truncation=True),
                              batched=True, remove_columns=["text"])
        meta_info = get_classification_meta(dataset, dataset_name + "_RE")
        re_datasets.append(SingleDataset(data=dataset, meta=meta_info))

    return re_datasets


###
# Classificaiton Tasks Data Loading
###
def get_all_classification_datasets(tokenizer, split="train") -> List[SingleDataset]:
    classification_datasets = []

    for dataset_loader in tqdm(get_all_dataloaders_for_task(Tasks.TEXT_CLASSIFICATION), desc="Preparing TEXT datasets"):
        dataset_name = Path(dataset_loader).with_suffix("").name

        try:
            dataset = datasets.load_dataset(str(dataset_loader), name=f"{dataset_name}_bigbio_text", split=split)
            meta_info = get_classification_meta(dataset, dataset_name + "_TEXT")
            dataset = dataset.map(lambda x: tokenizer(x["text"], max_length=512, truncation=True),
                                  batched=True, remove_columns=["text", "id", "document_id"])
            classification_datasets.append(SingleDataset(data=dataset, meta=meta_info))

        except (ValueError, ImportError) as err:
            print(f"Skipping {dataset_loader} because of {err}")
            continue

    return classification_datasets


class BioMuppet(pl.LightningModule):
    def __init__(
            self,
            transformer: str,
            lr: float,
            dataset_to_meta: Dict[str, DatasetMetaInformation],
            dropout=0.3,
            use_lr_scheduler: bool = True,
    ):
        super().__init__()


        self.dataset_to_meta = dataset_to_meta
        self.loss = nn.BCEWithLogitsLoss()

        self.use_lr_scheduler = use_lr_scheduler

        self.transformer = transformers.AutoModel.from_pretrained(transformer)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(transformer,
                                                                    use_fast=True)

        self.tokenizer.add_tokens(
            ["[HEAD-S]", "[HEAD-E]", "[TAIL-S]", "[TAIL-E]"], special_tokens=True
        )

        self.transformer.resize_token_embeddings(len(self.tokenizer))
        self.transformer_dropout = nn.Dropout(self.transformer.config.hidden_dropout_prob)
        self.non_transformer_dropout = nn.Dropout(dropout)

        self.dataset_to_out_layer = nn.ModuleDict()
        self.dataset_to_train_f1 = {}
        self.dataset_to_dev_f1 = {}

        for dataset, meta in dataset_to_meta.items():
            out_layer = self.dataset_to_out_layer[dataset] = nn.Linear(
                self.transformer.config.hidden_size, len(meta.label_to_id)
            )
            self.dataset_to_out_layer[dataset] = out_layer
            self.dataset_to_train_f1[dataset] = torchmetrics.F1Score(
                num_classes=len(meta.label_to_id)-1
            )
            self.dataset_to_dev_f1[dataset] = torchmetrics.F1Score(
                num_classes=len(meta.label_to_id)-1
            )

        self.lr = lr
        self.num_training_steps = None

    def collate_fn(self, data):
        meta: DatasetMetaInformation
        all_metas = []
        all_labels = []
        collator = transformers.DataCollatorWithPadding(self.tokenizer,
                                                        max_length=512)
        for i in data:
            all_metas.append(i.pop("meta"))
            all_labels.append(i.pop("labels"))
        batch = collator(data)
        batch["meta"] = ToableList(all_metas)
        batch["labels"] = ToableList(all_labels)

        return batch

    def forward(self, features):
        if "token_type_ids" in features:
            output = self.transformer(
                input_ids=features["input_ids"],
                token_type_ids=features["token_type_ids"],
                attention_mask=features["attention_mask"],
            )
        else:
            output = self.transformer(
                input_ids=features["input_ids"],
                attention_mask=features["attention_mask"],
            )

        batch_seq_emb = self.transformer_dropout(output.last_hidden_state)
        batch_loss = 0

        for seq_emb, labels, meta in zip(batch_seq_emb, features["labels"], features["meta"]):
            logits = self.dataset_to_out_layer[meta.name](seq_emb)
            batch_loss += TASK_TYPE_TO_LOSS[meta.type](logits, labels=labels, meta=meta)

        return MultitaskClassifierOutput(
            loss=batch_loss/len(logits),
            dataset_to_logits={},
        )

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)

        if self.lr_schedulers():
            self.lr_schedulers().step()

        self.log("train/loss", output.loss, prog_bar=True)
        return output.loss

    # def training_epoch_end(self, outputs) -> None:
    #     for dataset, train_f1 in self.dataset_to_train_f1.items():
    #         self.log(f"{dataset}/train/f1_epoch", train_f1, prog_bar=False)


    def configure_optimizers(self):
        assert self.num_training_steps > 0
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.use_lr_scheduler:
            schedule = transformers.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0.1 * self.num_training_steps,
                num_training_steps=self.num_training_steps,
            )

            return [optimizer], [schedule]
        else:
            return [optimizer]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and not args.overwrite_output_dir:
        print(f"Output dir {args.output_dir} already exists. Use --overwrite_output_dir to overwrite it.")
        sys.exit(1)


    tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-base")
    train_datasets = []
    train_datasets += get_all_re_datasets(tokenizer)
    train_datasets += get_all_classification_datasets(tokenizer)

    dataset_to_meta = {i.meta.name: i.meta for i in train_datasets}

    model = BioMuppet(
        transformer="michiyasunaga/BioLinkBERT-base",
        lr=3e-5,
        dataset_to_meta=dataset_to_meta,
    )

    checkpoint_callback = ModelCheckpoint(dirpath=args.output_dir, save_last=True,
                                          save_top_k=4, monitor="train/loss", every_n_train_steps=10000)
    mixed_train_instances = []
    for dataset in tqdm(train_datasets):
        mixed_train_instances.extend(dataset)

    train_loader = DataLoader(
        dataset=mixed_train_instances,
        collate_fn=model.collate_fn,
        batch_size=8,
    )

    logger = WandbLogger(project="biomuppet")
    trainer = pl.Trainer(max_epochs=1, gpus=4, strategy="ddp", precision=16,
                         callbacks=[checkpoint_callback], logger=logger, min_steps=100000, max_steps=100000)
    model.num_training_steps = len(train_loader) * trainer.max_epochs

    print(f"Training on {len(train_datasets)} datasets with a total of {len(mixed_train_instances)} training instances...")
    # Train the model
    trainer.fit(
        model=model, train_dataloaders=train_loader
    )

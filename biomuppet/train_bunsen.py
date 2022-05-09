import argparse
import copy
import dataclasses
import itertools
import math
import sys
from dataclasses import dataclass
import random
from collections import defaultdict
from pathlib import Path
import importlib

import numpy as np
import datasets
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm, trange
from typing import Dict, List, Optional, Tuple

import torch
import torchmetrics
import transformers
from torch.utils.data import Dataset, BatchSampler, DataLoader, DistributedSampler
import pytorch_lightning as pl
from torch import nn
from transformers.models.auto.tokenization_auto import AutoTokenizer

from bigbio.utils.constants import Tasks

from biomuppet.classification import get_all_classification_datasets
from biomuppet.coreference_resolution import get_all_coref_datasets
from biomuppet.relation_extraction import get_all_re_datasets

pl.seed_everything(42)

NUM_GPUS = 4
GRADIENT_ACCUMULATION = 1
BATCH_SIZE = 32
MAX_STEPS = 140000 * GRADIENT_ACCUMULATION

def print_average_task_mixing(dataloader, num_samples=1):
    avg_tasks_per_epoch = []
    for _ in trange(num_samples, desc="Calculating task mixing"):
        num_tasks_per_batch = []
        batches_per_gpu = [[] for _ in range(NUM_GPUS)]
        for idx_batch, batch in enumerate(dataloader):
            idx_gpu = idx_batch % NUM_GPUS
            batches_per_gpu[idx_gpu].append(batch)

        tasks = set()
        for i, batches in enumerate(zip(*batches_per_gpu)):
            for batch in batches:
                tasks.update(set(meta.name for meta in batch["meta"]))
            if i % GRADIENT_ACCUMULATION == 0:
                num_tasks_per_batch.append(len(tasks))
                tasks = set()
        avg_tasks_per_epoch.append(np.mean(num_tasks_per_batch))

    print(avg_tasks_per_epoch)


def classification_loss(logits, labels, meta):
    assert logits.ndim == 2
    cls_logit = logits[0]
    label_tensor = torch.zeros(len(meta.label_to_id))
    if labels:
        label = labels[0]
    else:
        label = "None"

    label_tensor[meta.label_to_id[label]] = 1
    loss = nn.CrossEntropyLoss()(cls_logit.unsqueeze(0), label_tensor.to(logits.device).unsqueeze(0))
    if len(meta.label_to_id) > 1:
        loss /= np.log(len(meta.label_to_id))

    return loss


def multilabel_classification_loss(logits, labels, meta):
    assert logits.ndim == 2
    cls_logit = logits[0]
    label_tensor = torch.zeros(len(meta.label_to_id))
    for label in labels:
        label_tensor[meta.label_to_id[label]] = 1
    loss = nn.BCEWithLogitsLoss()(cls_logit.unsqueeze(0), label_tensor.to(logits.device).unsqueeze(0))
    loss /= np.log(len(meta.label_to_id))

    return loss


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


@dataclass
class MultitaskClassifierOutput(transformers.file_utils.ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    dataset_to_logits: Dict[str, torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None



class BioMuppet(pl.LightningModule):
    def __init__(
            self,
            transformer: str,
            lr: float,
            dataset_to_meta: Dict[str, DatasetMetaInformation],
            dropout=0.1,
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
        self.dropout = nn.Dropout(dropout)

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
        data = copy.deepcopy(data)
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

        batch_seq_emb = self.dropout(output.last_hidden_state)
        batch_loss = 0

        for seq_emb, labels, meta in zip(batch_seq_emb, features["labels"], features["meta"]):
            logits = self.dataset_to_out_layer[meta.name](seq_emb)
            batch_loss += TASK_TYPE_TO_LOSS[meta.type](logits, labels=labels, meta=meta)

        return MultitaskClassifierOutput(
            loss=batch_loss,
            dataset_to_logits={},
        )

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)

        # if self.lr_schedulers():
        #     self.lr_schedulers().step()

        self.log("train/loss", output.loss, prog_bar=True, on_step=True, on_epoch=True)
        return output.loss

    # def training_epoch_end(self, outputs) -> None:
    #     for dataset, train_f1 in self.dataset_to_train_f1.items():
    #         self.log(f"{dataset}/train/f1_epoch", train_f1, prog_bar=False)


    def configure_optimizers(self):
        no_decay = ["bias", "gamma", "beta", "LayerNorm.bias", "LayerNorm.weight"]
        param_groups = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.Adam(param_groups, lr=self.lr)

        if self.use_lr_scheduler:
            schedule = transformers.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.025 * MAX_STEPS),
                num_training_steps=MAX_STEPS,
            )

            return [optimizer], [{
                "scheduler": schedule,
                "interval": "step",
                "frequency": 1
            }]
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
    train_datasets += get_all_re_datasets()
    train_datasets += get_all_classification_datasets()
    train_datasets += get_all_coref_datasets()
    for dataset in train_datasets:
        columns_to_remove = [column for column in dataset.data["train"].column_names if column != "labels"]
        dataset.data = dataset.data.map(lambda x: tokenizer(x["text"], max_length=512, truncation=True), batched=True, remove_columns=columns_to_remove)

    dataset_to_meta = {i.meta.name: i.meta for i in train_datasets}

    model = BioMuppet(
        transformer="michiyasunaga/BioLinkBERT-base",
        lr=3e-5,
        dataset_to_meta=dataset_to_meta,
        use_lr_scheduler=True
    )

    callbacks = []
    callbacks.append(ModelCheckpoint(dirpath=args.output_dir, every_n_train_steps=5000,
                                     save_top_k=-1))
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    mixed_train_instances = []
    for dataset in tqdm(train_datasets):
        mixed_train_instances.extend(dataset)

    train_loader = DataLoader(
        dataset=mixed_train_instances,
        collate_fn=model.collate_fn,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print_average_task_mixing(train_loader)

    logger = WandbLogger(project="biomuppet")
    trainer = pl.Trainer(gpus=NUM_GPUS, strategy="ddp", precision=16,
                         callbacks=callbacks,
                         accumulate_grad_batches=GRADIENT_ACCUMULATION,
                         logger=logger, max_steps=MAX_STEPS, gradient_clip_val=5)
    model.num_training_steps = MAX_STEPS

    print(f"Training on {len(train_datasets)} datasets with a total of {len(mixed_train_instances)} training instances...")
    # Train the model
    trainer.fit(
        model=model, train_dataloaders=train_loader
    )

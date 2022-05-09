import dataclasses
import itertools
import json
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import List
import importlib.util
from inspect import getmembers
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple, Union

import datasets
from datasets import DatasetDict
from flair.tokenization import SegtokSentenceSplitter, SpaceTokenizer
from tqdm import tqdm
import numpy as np
import pandas as pd
import glob
import re
import multiprocessing
from copy import deepcopy

import bigbio
from bigbio.utils.constants import Tasks

DEBUG = False
tokenizer = SpaceTokenizer()

# Skip non-english, local dataset, and problematic dataset
ignored_datasets = [
    'n2c2_2018_track2', 'n2c2_2018_track1', 'n2c2_2011', 'n2c2_2010',
    'n2c2_2009', 'n2c2_2008', 'n2c2_2006_smokers', 'n2c2_2006_deid',
    'psytar', 'swedish_medical_ner', 'quaero', 'pho_ner', 'ctebmsp',
    'codiesp', 'pubtator_central',
]

# Dataset to name & subset_id mapping for special cases datasets
dataset_to_name_subset_map = {
    'n2c2_2010': [('n2c2_2010_bigbio_kb', 'n2c2_2010')],
    'spl_adr_200db': [('spl_adr_200db_train_bigbio_kb', 'spl_adr_200db_train')],
    'pubtator_central' : [('pubtator_central_bigbio_kb','pubtator_central')],
    'mantra_gsc': [
        ('mantra_gsc_en_emea_bigbio_kb', 'mantra_gsc_en_EMEA'), 
        ('mantra_gsc_en_medline_bigbio_kb', 'mantra_gsc_en_Medline'),
        ('mantra_gsc_en_patents_bigbio_kb','mantra_gsc_en_Patents')
    ],
    'tmvar_v1': [('tmvar_v1_bigbio_kb','tmvar_v1')],
    'tmvar_v2': [('tmvar_v2_bigbio_kb','tmvar_v2')],
    'genetag': [
        ('genetaggold_bigbio_kb', 'genetaggold'),
        ('genetagcorrect_bigbio_kb', 'genetagcorrect')
    ],
    'chebi_nactem': [
        ('chebi_nactem_abstr_ann1_bigbio_kb', 'chebi_nactem_abstr_ann1'), 
        ('chebi_nactem_abstr_ann2_bigbio_kb', 'chebi_nactem_abstr_ann2'), 
        ('chebi_nactem_fullpaper_bigbio_kb', 'chebi_nactem_fullpaper')
    ],
    'dian_iber_eval': [('diann_iber_eval_en_bigbio_kb', 'diann_iber_eval_en')],
    'pubtator_central': [('pubtator_central_sample_bigbio_kb', 'pubtator_central')],
    'codiesp': [('codiesp_X_bigbio_kb', 'codiesp_x')],
    'muchmore': [('muchmore_en_bigbio_kb','muchmore_en')]
}

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

###
# NER Utils
###
def get_biodataset_metadata():
    biodataset_path = biodatasets_path = Path(bigbio.__file__).resolve().parents[1] / "biodatasets"
    datasets_meta = []

    for path in glob.glob(f'{biodataset_path}/*/*.py'):
        folder_path, file_path = path.replace('.py','').split('/')[-2:]
        if file_path != folder_path:
            # skip __init__.py and other (maybe) additional scripts
            continue

        lines = open(path).readlines()

        schemas = set()
        source_count, bigbio_count = 0, 0
        is_download_dataset = False
        line_start = 0
        regex = re.compile('[^a-zA-Z_0-9]')
        for i, line in enumerate(lines):
            if 'BigBioConfig(' in line:
                line_start = i
            elif 'schema=' in line: # Schema
                # check description (manually checked this rule before and it works fine to filter other use of the word `schema` :p)
                if 'description' not in lines[i-1]:
                    continue

                name = "?"
                for prev_line in lines[line_start:i]:
                    if 'name=' in prev_line:
                        name = prev_line.split('name=')[1].replace("f'","").replace('f"','')
                        name = regex.sub('', name)
                        break

                subset_id = "?"
                for next_line in lines[i:i+5]:
                    if 'subset_id=' in next_line:
                        subset_id = next_line.split('subset_id=')[1].replace("f'","").replace('f"','')
                        subset_id = regex.sub('', subset_id)
                        break

                schema = line.split('schema=')[1]
                if schema[0] == '"':
                    schemas.add((name, subset_id, schema[1:].split('"')[0]))
                else:
                    schema = schema.strip()[:-1].lower()
                    if 'bigbio' in schema or 'source' in schema:
                        schemas.add((name, subset_id, schema))

                if 'source' in schema:
                    source_count += 1
                else:
                    bigbio_count += 1
            elif 'download_and_extract' in line: # Downlaod
                is_download_dataset = True

        datasets_meta.append({
            'dataset_name': file_path,
            'has_source': 'source' in schemas,
            'has_bigbio': any(['bigbio' in schema for schema in schemas]),
            'has_bigbio_kb': any(['bigbio_kb' in schema for schema in schemas]),
            'has_bigbio_text': any(['bigbio_text' in schema for schema in schemas]),
            'has_bigbio_qa': any(['bigbio_qa' in schema for schema in schemas]),
            'has_bigbio_entailment': any(['bigbio_te' in schema for schema in schemas]),
            'has_bigbio_text2text': any(['bigbio_t2t' in schema for schema in schemas]),
            'has_bigbio_pairs': any(['bigbio_pairs' in schema for schema in schemas]),            
            'is_download_dataset': is_download_dataset,
            'source_schema_count': source_count,
            'bigbio_schema_count': bigbio_count,
            'schemas': list(schemas)
        })
    df = pd.DataFrame(datasets_meta)
    return df

def merge_discontiguous_entities(entities):
    # Break multiple entities into a single one
    for i, entity in enumerate(entities):
        if len(entity['offsets']) == 1:
            continue
            
        s_idx, e_idx = entity['offsets'][0]
        for offset in entity['offsets'][1:]:
            if offset[0] < s_idx:
                s_idx = offset[0]
            if offset[1] > e_idx:
                e_idx = offset[1]
                
        entities[i] = {
            'id': entities[i]['id'],
            'type': entities[i]['type'],
            'text': [' '.join(entities[i]['text'])],
            'offsets': [[s_idx, e_idx]],
            'normalized': []
        }
    return entities

class DpEntry(NamedTuple):
    position_end: int
    entity_count: int
    entity_lengths_sum: int
    last_entity: Optional[dict] 
    
def rearrange_overlapped_entities(entities):
    # Break nested entities into multiple non-overlapping entities. The code is adapted from: 
    # https://github.com/flairNLP/flair/blob/27e6c41dbfc540ae1302c1b400006a15e46575c0/flair/datasets/biomedical.py#L152
    # Uses dynamic programming approach to calculate maximum independent set in interval graph
    # with sum of all entity lengths as secondary key
    num_before = len(entities)
    
    dp_array = [DpEntry(position_end=0, entity_count=0, entity_lengths_sum=0, last_entity=None)]
    for entity in sorted(entities, key=lambda ent: (ent['offsets'][0][1] * 100) - ent['offsets'][0][0]):
        i = len(dp_array) - 1
        while dp_array[i].position_end > entity['offsets'][0][0]:
            i -= 1
        
        len_span = (entity['offsets'][0][1] - entity['offsets'][0][0])
        if dp_array[i].entity_count + 1 > dp_array[-1].entity_count or (
            dp_array[i].entity_count + 1 == dp_array[-1].entity_count
            and dp_array[i].entity_lengths_sum + len_span > dp_array[-1].entity_lengths_sum
        ):
            dp_array += [
                DpEntry(
                    position_end=entity['offsets'][0][1],
                    entity_count=dp_array[i].entity_count + 1,
                    entity_lengths_sum=dp_array[i].entity_lengths_sum + len_span,
                    last_entity=entity
                )
            ]
        else:
            dp_array += [dp_array[-1]]

    independent_entities = []
    p = dp_array[-1].position_end
    for dp_entry in dp_array[::-1]:
        if dp_entry.last_entity is None:
            break
        if dp_entry.position_end <= p:
            independent_entities += [dp_entry.last_entity]
            p = dp_entry.last_entity['offsets'][0][0]

    return independent_entities
            
    
def bigbio_ner_to_conll(sample):
    regex = re.compile('[^a-zA-Z_0-9\-]')                        
    
    # Sort passages & retrieve the ordered offsets
    sample['passages'] = sorted(sample['passages'], key=lambda pas: pas['offsets'][0][0])
    passage_offsets = list(map(lambda p: p['offsets'][0], sample['passages'])) # [(L1, R1), (L2, R2), ..., (Ln, Rn)]

    # Preprocess entity
    entities = rearrange_overlapped_entities(merge_discontiguous_entities(sample['entities']))
    
    # Generate CONLL formatted data
    conll_data = []
    passage = sample['passages'][0]['text'][0].replace('\t',' ').replace('\n',' ')
    p_idx, p_offset = 0, passage_offsets[0]
    for entity in sorted(entities, key=lambda e: e['offsets'][0][0]):
        # check entity offset & advance passage if needed
        s_idx, e_idx = entity['offsets'][0][0], entity['offsets'][0][1]
        while s_idx > p_offset[1]:
            # No entities on the passage, convert the rest of the passage to tokens without entity
            if len(passage) > 0:
                sentence = passage.replace('\t',' ').replace('\n',' ')
                for token in tokenizer.tokenize(sentence):
                    conll_data.append((token, 'O'))
            p_idx += 1
            passage = sample['passages'][p_idx]['text'][0].replace('\t',' ').replace('\n',' ')
            p_offset = passage_offsets[p_idx]

        # convert absolute index to relative index
        rs_idx, re_idx = s_idx - p_offset[0], e_idx - p_offset[0]
        
        # extract tokens without entity
        sentence = passage[:rs_idx].strip().replace('\t',' ').replace('\n',' ')
        for token in tokenizer.tokenize(sentence):
            conll_data.append((token, 'O'))
                                         
        # extract entity tokens
        label = passage[rs_idx:re_idx]
        for i, token in enumerate(tokenizer.tokenize(label)):
            conll_data.append((token, f"{'B' if i == 0 else 'I'}-{regex.sub('', entity['type'])}"))
                                         
        # extract entity words
        passage = passage[re_idx:]
        p_offset[0] += re_idx

    # No entity left, convert the rest of document to tokens without entity
    if len(passage) > 0 or p_idx < len(sample['passages']) - 1:
        while True:
            sentence = passage.replace('\t',' ').replace('\n',' ')
            for token in tokenizer.tokenize(sentence):
                conll_data.append((token, 'O'))
            p_idx += 1
            if p_idx < len(sample['passages']):                                         
                passage = sample['passages'][p_idx]['text'][0]                                         
            else:
                break                    

    sample['conll'] = conll_data
    return sample

def get_sequence_labelling_meta(dataset, name):
    label_to_idx = {"None": 0}
    for dset_split in dataset.keys():
        if len(dataset[dset_split]) == 0:
            continue

        for conll_data in dataset[dset_split]['conll']:
            for token, label in conll_data:
                if label not in label_to_idx:
                    label_to_idx[label] = len(label_to_idx)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    task_type = "sequence_labeling"

    return DatasetMetaInformation(
        id_to_label=idx_to_label,
        label_to_id=label_to_idx,
        type=task_type,
        name=name
    )

###
# Dataset Wrapper
###
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
        
        sts_datasets.append(SingleDataset(dataset, name=dataset_name + "_sts"))

        if DEBUG:
            break

    return sts_datasets

def get_all_ner_datasets() -> List[SingleDataset]:
    ner_datasets = []
    meta_df = get_biodataset_metadata()

    for idx, dataset_loader in enumerate(tqdm(
            get_all_dataloaders_for_task(Tasks.NAMED_ENTITY_RECOGNITION),
            desc="Preparing NER datasets",
    )):
        dataset_name = Path(dataset_loader).with_suffix("").name

        if dataset_name in ignored_datasets:
            continue

        if dataset_name in dataset_to_name_subset_map:
            for name, subset_id in dataset_to_name_subset_map[dataset_name]:
                try:
                    dataset = datasets.load_dataset(str(dataset_loader), name=name, subset_id=subset_id)
                    dataset = dataset.map(bigbio_ner_to_conll,
                        remove_columns=['passages', 'entities', 'events', 'coreferences', 'relations'],
                        load_from_cache_file=not DEBUG,
                        num_proc=multiprocessing.cpu_count() * 2
                    )
                    meta = get_sequence_labelling_meta(dataset, f"{name.replace('_bigbio_kb','')}_ner")
                    ner_datasets.append(SingleDataset(dataset, meta))
                except Exception as ve:
                    print(f"Skipping {dataset_loader} (name: {name}, subset_id:: {subset_id}) because of {ve}")            
        else:
            for name, subset_id, schema in meta_df.loc[meta_df['dataset_name'] == dataset_name, 'schemas'].values[0]:
                if 'bigbio_kb' not in schema:
                    continue
                try:
                    dataset = datasets.load_dataset(str(dataset_loader), name=name, subset_id=subset_id)
                    dataset = dataset.map(bigbio_ner_to_conll, 
                        remove_columns=['passages', 'entities', 'events', 'coreferences', 'relations'],
                        load_from_cache_file=not DEBUG,
                        num_proc=multiprocessing.cpu_count() * 2
                    )
                    meta = get_sequence_labelling_meta(dataset, f"{name.replace('_bigbio_kb','')}_ner")
                    ner_datasets.append(SingleDataset(dataset, meta))
                except Exception as ve:
                    print(f"Skipping {dataset_loader} (name: {name}, subset_id:: {subset_id}) because of {ve}")
        if DEBUG:
            if idx >= 3:
                break

    return ner_datasets

if __name__ == "__main__":
    re_datasets = get_all_re_datasets()
    coref_datasets = get_all_coref_datasets()
    classification_datasets = get_all_classification_datasets()
    ner_datasets = get_all_ner_datasets()
    sts_datasets = get_all_sts_datasets()

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
            train_valid = dataset.data["train"].train_test_split(test_size=0.1, seed=0)
            dataset.data = DatasetDict({
                "train": train_valid["train"],
                "valid": train_valid["test"],
            })

        ### Write train file
        with (out / dataset.name).with_suffix(".train").open("w", encoding="utf8") as f:
            for example in dataset.data["train"]:
                text = example["text"].strip().replace("\t", " ").replace("\n", " ")
                if not text:
                    continue
                label = "|".join(sorted(example["labels"]))
                if not label.strip():
                    label = "None"

                f.write(text + "\t" + label + "\n")

        ### Write validation file
        with (out / dataset.name).with_suffix(".valid").open("w", encoding="utf8") as f:
            for example in dataset.data["valid"]:
                text = example["text"].strip().replace("\t", " ").replace("\n", " ")
                if not text:
                    continue
                label = "|".join(sorted(example["labels"]))
                if not label.strip():
                    label = "None"

                f.write(text + "\t" + label + "\n")

    for dataset in tqdm(ner_datasets):
        if not "train" in dataset.data:
            continue
            
        print(f'Saving `{dataset.meta.name}`')
        config[dataset.meta.name] = {
            "train_data_path": str((out / dataset.meta.name).with_suffix(".train")),
            "validation_data_path": str((out / dataset.meta.name).with_suffix(".valid")),
            "word_idx": 0,
            "tasks": {
                dataset.meta.name: {
                    "column_idx": 1,
                    "task_type": "seq"
                }
            }
        }
        
        ### Generate validation split if not available            
        if not "valid" in dataset.data:
            train_valid = dataset.data["train"].train_test_split(test_size=0.1, seed=0)
            dataset.data = DatasetDict({
                "train": train_valid["train"],
                "valid": train_valid["test"],
            })            

        ### Write train file
        with (out / dataset.meta.name).with_suffix(".train").open("w") as f:
            for example in dataset.data["train"]:
                for word, label in example['conll']:
                    f.write(word + "\t" + label + "\n")
                f.write( "\n")

        ### Write validation file
        with (out / dataset.meta.name).with_suffix(".valid").open("w") as f:
            for example in dataset.data["valid"]:
                for word, label in example['conll']:
                    f.write(word + "\t" + label + "\n")
                f.write( "\n")
    
    for dataset in tqdm(sts_datasets):
        # WRONG, none of them are classification, except mqp (0/1 dissimilar/similar)
        config[dataset.name] = {
            "train_data_path": str((out / dataset.name).with_suffix(".train")),
            "validation_data_path": str((out / dataset.name).with_suffix(".valid")),
            "sent_idxs": [0,1],
            "tasks": {
                dataset.name: {
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
        
        dataset.data["train"].to_csv((out / dataset.name).with_suffix(".train"), sep="\t", index=None, header=None)
        dataset.data["valid"].to_csv((out / dataset.name).with_suffix(".valid"), sep="\t", index=None, header=None)

    ## Write Machamp config
    fname = "machamp/configs/bigbio_debug.json" if DEBUG else "machamp/configs/bigbio_full.json"
    with open(fname, "w") as f:
        json.dump(config, f, indent=1)
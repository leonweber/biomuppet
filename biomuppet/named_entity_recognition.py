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
from flair.tokenization import SpaceTokenizer
from tqdm import tqdm
import pandas as pd
import glob
import re
import multiprocessing

import bigbio
from bigbio.utils.constants import Tasks

from biomuppet.utils import DatasetMetaInformation, SingleDataset, get_all_dataloaders_for_task, split_sentences, DEBUG

# Define tokenizer globally
tokenizer = SpaceTokenizer()

DEBUG = False

# Skip non-english, local dataset, and problematic dataset
ignored_datasets = [
    'n2c2_2018_track2', 'n2c2_2018_track1', 'n2c2_2011', 'n2c2_2010',
    'n2c2_2009', 'n2c2_2008', 'n2c2_2006_smokers', 'n2c2_2006_deid',
    'psytar', 'swedish_medical_ner', 'quaero', 'pho_ner', 'ctebmsp',
    'codiesp', 'pubtator_central', 'cord_ner'
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
        ('genetaggold_bigbio_kb', 'genetaggold')
    ],
    'chebi_nactem': [
        ('chebi_nactem_abstr_ann1_bigbio_kb', 'chebi_nactem_abstr_ann1'), 
        ('chebi_nactem_fullpaper_bigbio_kb', 'chebi_nactem_fullpaper')
    ],
    'cellfinder': [('cellfinder_bigbio_kb','cellfinder')],
    'bioscope': [
        ('bioscope_abstracts_bigbio_kb','bioscope_abstracts'),
        ('bioscope_papers_bigbio_kb','bioscope_papers'),
    ],
    'diann_iber_eval': [('diann_iber_eval_en_bigbio_kb', 'diann_iber_eval_en')],
    # 'pubtator_central': [('pubtator_central_sample_bigbio_kb', 'pubtator_central')],
    # 'codiesp': [('codiesp_X_bigbio_kb', 'codiesp_x')],
    'muchmore': [('muchmore_en_bigbio_kb','muchmore_en')]
}

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
        name=name,
        entropy=None
    )

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
            conll_data.append(('\n', '\n')) # Chunk data between sentence
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
                    dataset = dataset.map(split_sentences, num_proc=multiprocessing.cpu_count() * 2)
                    dataset = dataset.map(bigbio_ner_to_conll,
                        remove_columns=['passages', 'entities', 'events', 'coreferences', 'relations'],
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
                    dataset = dataset.map(split_sentences, num_proc=multiprocessing.cpu_count() * 2)
                    dataset = dataset.map(bigbio_ner_to_conll, 
                        remove_columns=['passages', 'entities', 'events', 'coreferences', 'relations'],
                        num_proc=multiprocessing.cpu_count() * 2
                    )
                    meta = get_sequence_labelling_meta(dataset, f"{name.replace('_bigbio_kb','')}_ner")
                    ner_datasets.append(SingleDataset(dataset, meta))
                except Exception as ve:
                    print(f"Skipping {dataset_loader} (name: {name}, subset_id:: {subset_id}) because of {ve}")
        if DEBUG and len(ner_datasets) >= 5:
            break

    return ner_datasets

if __name__ == '__main__':
    ner_datasets = get_all_ner_datasets()
    config = {}
    out = Path("machamp/data/bigbio/named_entity_recognition")
    out.mkdir(exist_ok=True, parents=True)

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
        if not "validation" in dataset.data:
            train_valid = dataset.data["train"].train_test_split(test_size=0.1, seed=0)
            dataset.data = DatasetDict({
                "train": train_valid["train"],
                "validation": train_valid["test"],
            })

            ### Write train file
        with (out / dataset.meta.name).with_suffix(".train").open("w") as f:
            for example in dataset.data["train"]:
                for word, label in example['conll']:
                    f.write(word + "\t" + label + "\n")
                f.write( "\n")

        ### Write validation file
        with (out / dataset.meta.name).with_suffix(".valid").open("w") as f:
            for example in dataset.data["validation"]:
                for word, label in example['conll']:
                    f.write(word + "\t" + label + "\n")
                f.write( "\n")

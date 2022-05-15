"""Get overlap between bigbio MACHAMP dataset + BLURB

I AM USING LINKBERT's BLURB DATASET AS OPPOSED TO PRE-PROCESSING FROM SCRATCH.

BLURB_datasets == dict of BLURB dataset name + reported BLURB size from microsoft table
BLURB2BB == dict of BLURB dataset name to the bigbio dataset to load
bb_configs == name of bigbio dataset and configs to load. NOTE PUBMED HAS MULTIPLE

Compare OVERLAPS:
    BLURB Train < -- > Test
    BLURB Train < -- > Train

TODOs:
Switch to logging
"""
import re
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from typing import List, Dict, Set
from constants import BLURB_datasets, text_pairs, ner_examples, task_mapping
import gzip 
import pickle as pkl

# ------------------------- #
# BLURB dataset fxns
# ------------------------- #
def get_name(dpath: Path) -> str:
    """Return name from a datapath object"""
    return dpath.__str__().split("/")[-1]


def get_blurb_size(data: DatasetDict) -> List[int]:
    """Compute number of examples in each key"""
    return [data[key].num_rows for key in data.keys()]

def get_linkbert_blurb_preprocessed_data(datapath: Path) -> DatasetDict:
    """Gets pre-processed data from LINKBERT.
    Uses HuggingFace's datasets to load.
    """
    files = [i.__str__().split("/")[-1] for i in datapath.glob("*.json")]
    data_files = {item.split(".json")[0] : item for item in files}
    return load_dataset(datapath.__str__(), data_files=data_files)

# TODO - clean string here?
def get_blurb_sentences(data: DatasetDict, name: str) -> Dict[str, Set[str]]:
    """Get sentences from each training split
    NOTE:
        text pairs - sentence 1 + sentence 2 are added as separate items.
        This means that every example contributes 2 "sentences" in each set.

        NER tokens - are joined by white space and stripped of excessive white spaces.

        else - everything else has each sentence added 
    """
    if name in text_pairs:
        output = {key: set(data[key]["sentence1"]) for key in data.keys()}
        output = {key: output[key].union(set(data[key]["sentence2"])) for key in data.keys()}
    
    elif name in ner_examples:
        output = {key: [" ".join(i) for i in data[key]["tokens"]] for key in data.keys()}
        output = {key: set([clean_string(s) for s in output[key]]) for key in data.keys()}
        
    else:
        output = {key: set(data[key]["sentence"]) for key in data.keys()}
    return output

def clean_string(s: str) -> str:
    """Fix casing/extra white spacing/regex issues"""
    s = re.sub(' +', ' ', s)
    s = re.sub(r'\s([?.,!"](?:\s|$))', r'\1', s)
    return s


def collect_blurb_data(data_paths: List[Path]) -> Dict[str, Set[str]]:
    """Collects ALL BLURB train/val/test splits.
    Creates dict holding each data split.

    :param data_paths: Path locs for all BLURB datasets 
    """
    blurb_text_pairs = {"train": set(), "dev": set(), "test": set()}
    blurb_ner = {"train": set(), "dev": set(), "test": set()}
    blurb = {"train": set(), "dev": set(), "test": set()}

    for dpath in data_paths:
        name = get_name(dpath)

        print("Dataset", name)
        bdata = get_linkbert_blurb_preprocessed_data(dpath)
        blurb_sentences = get_blurb_sentences(bdata, name)

        if name in text_pairs:
            for key in blurb_sentences.keys():
                blurb_text_pairs[key] = blurb_text_pairs[key].union(blurb_sentences[key])
        elif name in ner_examples:
            for key in blurb_sentences.keys():
                blurb_ner[key] = blurb_ner[key].union(blurb_sentences[key])
        else:
            for key in blurb_sentences.keys():
                blurb[key] = blurb[key].union(blurb_sentences[key])

    return blurb, blurb_ner, blurb_text_pairs

# ------------------------- #
# Machamp pre-processed data fxns
# ------------------------- #

def get_machamp_datasetname(fname: Path, ext: str):
    """Get dataset name to split on"""
    return fname.__str__().split("/")[-1].split(".")[0].split(ext)[0]

# TODO: maybe avoid pandas with just simple string parsing

def parse_machamp_ner(filename: Path) -> Set[str]:
    """Parse token-based tasks"""
    sents = set()
    line = []
    
    with open(filename, "r") as f:
        x = f.readlines()

    for l in x:
        if l == "\n":
            sents.add(" ".join(line))
            line = []
        else:
            line.append(l.split("\t")[0])
            
    return sents

def collect_machamp_data(tasks: List[Path], split: str) -> Dict[str, Set[str]]:
    """Given a machamp task, construct a set of all sentences from all datasets in it
    """
    machamp_data = {}

    non_ner_qa_tasks = [t for t in tasks if "qa" not in t.__str__() and "named_entity" not in t.__str__()]
    ner_tasks = [t for t in tasks if "named_entity" in t.__str__()]
    qa_tasks = [t for t in tasks if "qa" in t.__str__()]

    for tk in non_ner_qa_tasks:
        tk_name = tk.__str__().split("/")[-1]
        ext = task_mapping[tk_name]
        datapaths = tk.glob("*." + split)

        for fname in datapaths:

            dset_name = get_machamp_datasetname(fname, ext)

            if dset_name not in machamp_data.keys():
                machamp_data.update({dset_name: set()})

            try:
                x = pd.read_csv(fname, sep="\t", header=None) # Two columns, space separated
                machamp_data[dset_name] = machamp_data[dset_name].union(set(x.iloc[:, 0].tolist()))
                
            except pd.errors.EmptyDataError:
                print("Empty dataset with dataset", fname)
            except pd.errors.ParserError:
                print("Issue parsing with", fname)

    # Named entity recognition needs to be grouped by new-lines
    for tk in ner_tasks:
        tk_name = tk.__str__().split("/")[-1]
        ext = task_mapping[tk_name]
        datapaths = tk.glob("*." + split)

        for fname in datapaths:

            dset_name = get_machamp_datasetname(fname, ext)

            if dset_name not in machamp_data.keys():
                machamp_data.update({dset_name: set()})

            try:
                x = parse_machamp_ner(fname)
                machamp_data[dset_name] = machamp_data[dset_name].union(x)
                
            except pd.errors.EmptyDataError:
                print("Empty dataset with dataset", fname)
            except pd.errors.ParserError:
                print("Issue parsing with", fname)

    # Split QA tasks with both _CLF/_SEQ type tokens
    tk_name = qa_tasks[0].__str__().split("/")[-1]
    for ext in task_mapping[tk_name]:
        datapaths = qa_tasks[0].glob("*." + split)
        datapaths = [i for i in datapaths if ext in i.__str__()]
        
        for fname in datapaths:

            dset_name = get_machamp_datasetname(fname, ext)
            
            if dset_name not in machamp_data.keys():
                machamp_data.update({dset_name: set()})

            try:
                x = pd.read_csv(fname, sep="\t", header=None) # Two columns, space separated
                machamp_data[dset_name] = machamp_data[dset_name].union(set(x.iloc[:, 0].tolist()))
                
            except pd.errors.EmptyDataError:
                print("Empty dataset with dataset", fname)
            except pd.errors.ParserError:
                print("Issue parsing with", fname)

    return machamp_data


def compute_overlap(blurb: Set[str], bbio: Set[str]) -> Set[str]:
    """Given 2 datasplits, compute set overlap between them"""
    return blurb.intersection(bbio)

if __name__ == "__main__":
    data_dir = Path(Path(__file__).__str__().split('/')[0]) / "data"
    machamp_dir = Path(__file__).parents[1] / "machamp/data/bigbio"

    data_paths = list((data_dir / "seqcls").glob("*"))
    data_paths += list((data_dir / "tokcls").glob("*"))

    blurb, blurb_ner, blurb_text_pairs = collect_blurb_data(data_paths)

    # Get MACHAMP training data
    tasks = list(machamp_dir.glob("*/"))
    tasks = [i for i in tasks if ".git" not in i.__str__()]

    # For each task, compute the set of terms

    machamp_train = collect_machamp_data(tasks, "train")
    machamp_val = collect_machamp_data(tasks, "valid")

    print("Saving Linkbert Data")
    with gzip.open("linkbert_blurb.gz.pkl", "wb") as f:
        pkl.dump(blurb, f)

    with gzip.open("linkbert_blurb_ner.gz.pkl", "wb") as f:
        pkl.dump(blurb_ner, f)

    with gzip.open("linkbert_blurb_text_pairs.gz.pkl", "wb") as f:
        pkl.dump(blurb_text_pairs, f)

    print("Saving machamp train")
    with gzip.open("machamp_train.gz.pkl", "wb") as f:
        pkl.dump(machamp_train, f)

    print("Saving machamp validation")
    with gzip.open("machamp_val.gz.pkl", "wb") as f:
        pkl.dump(machamp_val, f)


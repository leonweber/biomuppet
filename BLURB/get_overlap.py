"""Get overlap between bigbio + BLURB

I AM USING LINKBERT's BLURB DATASET AS OPPOSED TO PRE-PROCESSING FROM SCRATCH.

Pubmed QA is really big, hence has several split configs.

BLURB_datasets == dict of BLURB dataset name + reported BLURB size from microsoft table
BLURB2BB == dict of BLURB dataset name to the bigbio dataset to load
bb_configs == name of bigbio dataset and configs to load. NOTE PUBMED HAS MULTIPLE

Compare OVERLAPS:
    BLURB Train < -- > Test
    BLURB Train < -- > Train

Switch to logging
"""
import bigbio
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from typing import List, Dict, Set
from nltk.tokenize import sent_tokenize
from constants import BLURB_datasets, text_pairs, ner_examples

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


def get_blurb_sentences(data: DatasetDict, name: str) -> Dict[str, Set[str]]:
    """Get sentences from each training split"""
    if name in text_pairs:
        output = {key: set(data[key]["sentence1"]) for key in data.keys()}
        output = {key: output[key].union(set(data[key]["sentence2"])) for key in data.keys()}
    
    elif name in ner_examples:
        output = {key: data[key]["tokens"] for key in data.keys()}
        output = {key: set([j for k in output[key] for j in k]) for key in output.keys()}
        
    else:
        output = {key: set(data[key]["sentence"]) for key in data.keys()}
    return output


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


# TODO: maybe avoid pandas with just simple string parsing
def collect_machamp_data(datapath: Path, split: str) -> Dict[str, Set[str]]:
    """Given a machamp task, construct a set of all sentences from all datasets in it
    """
    sents = set()
    for fname in datapath.glob("*." + split):
        try:
            x = pd.read_csv(fname, sep="\t", header=None) # Two columns, space separated
            sents = sents.union(set(x.iloc[:, 0].tolist()))
        except pd.errors.EmptyDataError:
            print("Issue with dataset", fname)

    return {split: sents}


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
    tasks = [i for i in tasks if ".gitignore" not in i.__str__()]

    # For each task, compute the set of terms
    machamp_train = {}
    machamp_val = {}

    for tk in tasks:
        tk_name = tk.__str__().split("/")[-1]
        print("Computing task = ", tk_name)
        machamp_train.update({tk_name: collect_machamp_data(tk, "train")})
        machamp_val.update({tk_name: collect_machamp_data(tk, "val")})



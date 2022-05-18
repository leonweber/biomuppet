import gzip
import pickle as pkl
import re
from typing import List, Set
from time import time
from tqdm import tqdm

import gzip as gz
import pickle as pkl

# Should I omit stop words?
#from nltk.corpus import stopwords
#stop_words = stopwords.words('english')

def clean_string(s: str) -> str:
    """Fix casing/extra white spacing/regex issues"""
    s = re.sub(' +', ' ', s)
    s = re.sub(r'\s([?.,!"](?:\s|$))', r'\1', s)
    s = s.lower()
    return s

def create_clean_tokens(s: str) -> Set[str]:
    """Given a str input, tokenize + clean string"""
    tokens = s.split(" ")
    tokens = [clean_string(tok) for tok in tokens]
    return set(tokens)


def tokenize_inputs(inp_set: List[str]) -> List[Set[str]]:
    """ Intake a set of strings and compute them into sets of tokens
    """
    tmp = [create_clean_tokens(i) for i in inp_set]
    return tmp

def compare_splits(
    split_blurb: Set[str],
    split_machamp: Set[str],
    thresh=0.8,
    ):
    """For each example, return the number of examples that have
    over 80% overlap with an example in the BLURB dataset, using the length of the unique tokens in the BLURB example.

    :param split_blurb: Set of blurb Examples
    :param split_machamp: Set of Machamp examples
    :param thresh: % of tokens that overlap between blurb/machamp
    """
    split_blurb = list(split_blurb)
    split_machamp = list(split_machamp)

    blurb_toks = tokenize_inputs(split_blurb)
    machamp_toks = tokenize_inputs(split_machamp)
    
    overlaps = []
    for idx, machamp_ex in enumerate(tqdm(machamp_toks, desc="MACHAMP ex")):
        for blurb_ex in blurb_toks:
            overlap_pct = len(machamp_ex.intersection(blurb_ex)) / len(blurb_ex)
            if overlap_pct >= thresh:
                overlaps.append(split_machamp[idx])
    
    return overlaps



if __name__ == "__main__":

    print("Loading Data")
    with gzip.open("linkbert_blurb.gz.pkl", "rb") as f:
        blurb = pkl.load(f)

    with gzip.open("linkbert_blurb_ner.gz.pkl", "rb") as f:
        blurb_ner = pkl.load(f)

    with gzip.open("linkbert_blurb_text_pairs.gz.pkl", "rb") as f:
        blurb_text_pairs = pkl.load(f)

    with gzip.open("machamp_train.gz.pkl", "rb") as f:
        machamp_train = pkl.load(f)

    with gzip.open("machamp_val.gz.pkl", "rb") as f:
        machamp_val = pkl.load(f)


    print("Beginning Overlap comparison")
    #  ------------------------------------------------ #
    for dsplit in ["train", "dev", "test"]:
        for dset in machamp_train:
            print("Dataset = ", dset)
            overlap = compare_splits(blurb[dsplit], machamp_train[dset])
            overlap_ner = compare_splits(blurb_ner[dsplit], machamp_train[dset])
            overlap_pairs = compare_splits(blurb_text_pairs[dsplit], machamp_train[dset])

            print(
                "BLURB " + dsplit + " v. Machamp Train; " + dset  + "\n",
                len(overlap),
                "/", len(blurb[dsplit]), " \n",
            )

            print(
                "BLURB NER" + dsplit + " v. Machamp Train; " + dset  + "\n",
                len(overlap_ner),
                "/", len(blurb_ner[dsplit]), " (NER) \n",
            )

            print(
                "BLURB Text Pairs " + dsplit + " v. Machamp Train; " + dset  + "\n",
                len(overlap_pairs),
                "/", len(blurb_text_pairs[dsplit]), " (text pairs) \n",
            )

            print("Saving data")
            with gz.open(dset + "_train_blurb_" + dsplit  + ".pkl.gz", "wb") as f:
                pkl.dump(overlap, f)

            with gz.open(dset + "_train_blurb_" + dsplit  + "_ner.pkl.gz", "wb") as f:
                pkl.dump(overlap_ner, f)

            with gz.open(dset + "_train_blurb_" + dsplit  + "_textpairs.pkl.gz", "wb") as f:
                pkl.dump(overlap_pairs, f)


    for dsplit in ["train", "dev", "test"]:
        for dset in machamp_val:
            overlap = compare_splits(blurb[dsplit], machamp_val[dset])
            overlap_ner = compare_splits(blurb_ner[dsplit], machamp_val[dset])
            overlap_pairs = compare_splits(blurb_text_pairs[dsplit], machamp_val[dset])

            print(
                "BLURB " + dsplit + " v. Machamp Val; " + dset  + "\n",
                len(overlap),
                "/", len(blurb[dsplit]), " \n",
            )

            print(
                "BLURB NER" + dsplit + " v. Machamp Val; " + dset  + "\n",
                len(overlap_ner),
                "/", len(blurb_ner[dsplit]), " (NER) \n",
            )

            print(
                "BLURB Text Pairs " + dsplit + " v. Machamp Val; " + dset  + "\n",
                len(overlap_pairs),
                "/", len(blurb_text_pairs[dsplit]), " (text pairs) \n",
            )

            print("Saving data")
            with gz.open(dset + "_val_blurb_" + dsplit  + ".pkl.gz", "wb") as f:
                pkl.dump(overlap, f)

            with gz.open(dset + "_val_blurb_" + dsplit  + "_ner.pkl.gz", "wb") as f:
                pkl.dump(overlap_ner, f)

            with gz.open(dset + "_val_blurb_" + dsplit  + "_textpairs.pkl.gz", "wb") as f:
                pkl.dump(overlap_pairs, f)

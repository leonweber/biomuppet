from calendar import c
import gzip
import imp
import pickle as pkl
import re
from typing import List, Set
import argparse
from time import time
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import os
import torch
from torch import nn 
import numpy as np

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
    tokens = " ".join([clean_string(tok) for tok in tokens])
    return tokens


def tokenize_inputs(inp_set: Set[str]) -> List[Set[str]]:
    """ Intake a set of strings and compute them into sets of tokens
    """
    tmp = [create_clean_tokens(i) for i in tqdm(list(inp_set))]
    return tmp


def compute_overlap(
    blurbset: List[str],
    machampset: List[str],
    thresh=0.8
    ):
    """Compute the overlap between blurb + machamp

    This computes a TFIDF vectorizer on the BLURB dataset (first)
    and then computes which "sentences" in the MACHAMP dataset
    have highest cosine-similarity (overlap)

    - Cleans tokens (marginal spacing + punctuation + lower case) for each set
    - Creates TFIDF vectorizer on BLURB input set
    - Transforms machamp set via TFIDF from BLURB
    - Computes matrix dot product between blurb TDIDF matrix * machamp matrix

    """
    bset = list(blurbset)
    mset = list(machampset)

    btoks = tokenize_inputs(blurbset)
    mtoks = tokenize_inputs(machampset)


    tfmodel = TfidfVectorizer()

    btok_mat = tfmodel.fit_transform(btoks).toarray()

    dataloader = torch.utils.data.DataLoader(mtoks, batch_size=1000, shuffle=False, num_workers=0)
    result = []
    offset = 0
    for batch in tqdm(dataloader):
        mtok_mat = tfmodel.transform(batch)
        sims = cosine_similarity(mtok_mat, btok_mat)
        max_sims = sims.max(axis=1)
        for i in np.where(max_sims > thresh)[0]:
            result.append((max_sims[i], mset[i+offset]))
        offset += len(batch)

    return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--worldsize", type=int, default=1)
    args = parser.parse_args()

    print("Loading Data")
    with gzip.open("save_blurb_by_dataset/blurb_train.gz.pkl", "rb") as f:
        blurb_train = pkl.load(f)

    with gzip.open("save_blurb_by_dataset/blurb_dev.gz.pkl", "rb") as f:
        blurb_dev = pkl.load(f)

    with gzip.open("save_blurb_by_dataset/blurb_test.gz.pkl", "rb") as f:
        blurb_test = pkl.load(f)

    with gzip.open("save_blurb_by_dataset/machamp_trainvalid.gz.pkl", "rb") as f:
        machamp_splits = pkl.load(f)

    
    blurb = {"train": blurb_train, "valid": blurb_dev, "test": blurb_test}

    if not os.path.exists("results"):
        os.mkdir("results")

    for dsplit in ["train", "valid"]:
        machamp_data = machamp_splits[dsplit]

        for msplit in blurb.keys():
            bdata = blurb[msplit]

            for i, dset in tqdm(enumerate(sorted(bdata.keys()))):

                if i % args.worldsize != args.rank:
                    continue

                print("BLURB-" + msplit + "; Machamp=" + dsplit + "; Dset=", dset)

                # #write overlap to file if it doesn't exist
                fname = "results/overlap_machamp_" + dsplit + "_blurb_" + msplit + "_" + dset + ".txt"
                if not os.path.exists(fname):
                    with open(fname, "w") as f:
                        for overlap, sent in compute_overlap(bdata[dset], machamp_data):
                            f.write(str(overlap) + "\t" + sent + "\n")
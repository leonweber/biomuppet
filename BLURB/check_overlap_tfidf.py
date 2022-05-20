import gzip
import pickle as pkl
import re
from typing import List, Set
from time import time
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    tmp = [create_clean_tokens(i) for i in list(inp_set)]
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

    btok_mat = tfmodel.fit_transform(btoks)
    mtok_mat = tfmodel.transform(mtoks)

    # TODO: LEON
    # The cosine_similarity fxn will fail here due to memory issues
    # we can home cook a cos-sim, but i wonder if a matrix math is sufficient

    # Compute the max overlap between the 2
    t0 = time()
    #overlap = cosine_similarity(btok_mat, mtok_mat) 
    overlap = btok_mat * mtok_mat.T 
    
    print(time() - t0, "Elapsed time")

    # TODO: LEON
    # I needed to write a loop as just a overlap.max(axis=0) will collapse due to size
    # Ideally here, you just need to ask if, for an entry in the machamp dataset
    # who the largest 

    # Ideally you want to do:
    #overlap.max(axis=1) > 0.8 (find any machamp entry with the largest BLURB overlap, if the blurb cos-sim is over let's say 0.8, then it has a high similarity to a sentence in BLURB
    # Collapse over all blurb indices 
    # The matrix overlap is N_examples_in_blurb x N_examples_in_machamp
    overlap_sents = []
    for idx in tqdm(range(mtok_mat.shape[0])):
        if overlap[:, idx].max() >= thresh:
            overlap_sents.append(mset[idx])

    return overlap, overlap_sents

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


    machamp_splits = {"train": machamp_train, "val": machamp_val}

    for dsplit in ["train", "dev", "test"]:

        for msplit in machamp_splits.keys():
            
            machamp_data = machamp_splits[msplit]

            for dset in machamp_data.keys():
                print("BLURB-" + dsplit + "; Machamp=" + msplit + "; Dset=", dset)

                #overlap_blurb = compute_overlap(blurb[dsplit], machamp_data[dset])
                #overlap_blurb_ner = compute_overlap(blurb_ner[dsplit], machamp_data[dset])
                #overlap_blurb_text_pairs = compute_overlap(blurb_text_pairs[dsplit], machamp_data[dset])

                #print("BLURB overlap =", )
                #print("BLURB (NER) overlap =", )
                #print("BLURB (text pairs) overlap =", )
            
            
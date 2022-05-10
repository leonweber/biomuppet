"""Get overlap between bigbio + BLURB
"""
import bigbio
from pathlib import Path
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from typing import List, Dict, Set
from nltk.tokenize import sent_tokenize

# Train/Dev/Test Values from: https://microsoft.github.io/BLURB/tasks.html#dataset_chemprot
BLURB_datasets = {
    "BC5CDR-chem_hf": [5203, 5347, 5385],
    "BC5CDR-disease_hf": [4182, 4244, 4424],
    "NCBI-disease_hf": [5134, 787, 960],
    "BC2GM_hf": [15197, 3061, 6325],
    "JNLPBA_hf": [46750, 4551, 8662],
    "ebmnlp_hf": [339167, 85321, 16364],
    "chemprot_hf": [18035, 11268, 15745],
    "DDI_hf": [25296, 2496, 5716],
    "GAD_hf": [4261, 535, 534],
    "BIOSSES_hf": [64, 16, 20],
    "hoc_hf": [1295, 186, 371],
    "HoC_hf": [1295, 186, 371],
    "pubmedqa_hf": [450, 50, 500],
    "bioasq_hf": [670, 75, 140],  # Task 7b
}

# BLURB names <-> BigBio
BLURB2BB = {
    "BC5CDR-chem_hf": [5203, 5347, 5385],
    "BC5CDR-disease_hf": [4182, 4244, 4424],
    "NCBI-disease_hf": [5134, 787, 960],
    "BC2GM_hf": [15197, 3061, 6325],
    "JNLPBA_hf": [46750, 4551, 8662],
    "ebmnlp_hf": [339167, 85321, 16364],
    "chemprot_hf": [18035, 11268, 15745],
    "DDI_hf": [25296, 2496, 5716],
    "GAD_hf": [4261, 535, 534],
    "BIOSSES_hf": [64, 16, 20],
    "hoc_hf": [1295, 186, 371],
    "HoC_hf": [1295, 186, 371],
    "pubmedqa_hf": [450, 50, 500],
    "bioasq_hf": [670, 75, 140],  # Task 7b  
}

BLURB2BB = {
    "BC5CDR-chem_hf": None,
    "BC5CDR-disease_hf": "bc5cdr",
    "NCBI-disease_hf": "ncbi_disease",
    "BC2GM_hf": "gnormplus",
    "JNLPBA_hf": None,
    "ebmnlp_hf": "ebm_pico",
    "chemprot_hf": "chemprot",
    "DDI_hf": "ddi_corpus",
    "GAD_hf": None,
    "BIOSSES_hf": "biosses",
    "hoc_hf": "hallmarks_of_cancer",
    "HoC_hf": "hallmarks_of_cancer",
    "pubmedqa_hf": "pubmed_qa",
    "bioasq_hf": "bioasq_task_b",
}

#bb_configs = {
#    "bc5cdr": ["bc5cdr_bigbio_kb"],
#    "gnormplus": ["gnormplus_bigbio_kb"]
#    "ncbi_disease": ["ncbi_disease_bigbio_kb"],
#    "ebm_pico": ["ebm_pico_bigbio_kb"],
#    "chemprot": ["chemprot_bigbio_kb"], # "chemprot_shared_task_eval_source", 
#    "ddi_corpus": ["ddi_corpus_bigbio_kb"],
#    "biosses": ["biosses_bigbio_pairs"],
#    "hallmarks_of_cancer": ["hallmarks_of_cancer_bigbio_text"],
#    "pubmed_qa": ["pubmed_qa_labeled_fold" + str(i) for i in range(1, 11)],
#    "bioasq_task_b": ["bioasq_7b_bigbio_qa"],
#}

# ------------------------- #
# BLURB dataset fxns
# ------------------------- #

def get_link_preprocessed_data(datapath: Path) -> DatasetDict:
    """Gets pre-processed data from LINKBERT.
    Uses HuggingFace's datasets to load.
    """
    files = [i.__str__().split("/")[-1] for i in datapath.glob("*.json")]
    data_files = {item.split(".json")[0] : item for item in files}
    return load_dataset(datapath.__str__(), data_files=data_files)

def get_name(dpath: Path) -> str:
    """Return name from a datapath object"""
    return dpath.__str__().split("/")[-1]

def get_blurb_size(data: DatasetDict) -> List[int]:
    """Compute number of examples in each key"""
    return [data[key].num_rows for key in data.keys()]

def get_blurb_sentences(data: DatasetDict) -> Dict[str, Set[str]]:
    """Get sentences from each training split"""
    return {key: set(data[key]["sentence"]) for key in data.keys()}

# ------------------------- #
# BigBio dataset fxns
# ------------------------- #
def get_bigbio_dataset(dpath: Path, name: str) -> List[DatasetDict]:
    """Retrieve BigBio Dataset, given name"""
    bbname = BLURB2BB[name] 

    if bbname is None:
        return None
    else:
        dsetname = bbname + ".py"
        dpath = dpath / bbname / dsetname
        return [load_dataset(dpath.__str__(), name=c) for c in bb_configs[bbname]]


def split_sentences(data: List[List[Dict[str, str]]]) -> Set[str]:
    """Given passages in a dataset, split into sentences

    data: A given data split "passages" argument in the BB schema (ex: biodata["train"]["passages"])
    """
    sents = set()
    if len(data):
        for psg in data:
            if len(psg):
                s = " ".join([" ".join(i["text"]) for i in psg])  # Smush all sentences together
                sents = sents.union(set(sent_tokenize(s)))
    return sents

def get_bigbio_sentences():
    """Return key: sentence"""
    pass


if __name__ == "__main__":
    curr_dirr = Path(__file__).__str__().split('/')[0]
    data_dir = Path(curr_dirr) / "data"

    bigbio_path = Path("/home/natasha/Projects/hfbiomed/biomedical/biodatasets")

    data_paths = list((data_dir / "seqcls").glob("*"))
    data_paths += list((data_dir / "tokcls").glob("*"))

    for dpath in data_paths:
        name = get_name(dpath)
        print("Dataset", name)
        bdata = get_link_preprocessed_data(dpath)
        assert(get_blurb_size(bdata) == BLURB_datasets[name], "Issue with " + name)

        #bbiodata = get_bigbio_dataset(bigbio_path, name)
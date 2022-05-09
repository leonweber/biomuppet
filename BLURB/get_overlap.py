"""Get overlap between bigbio + BLURB
"""

from pathlib import Path
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from typing import List

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
    return [data[key].num_rows for key in data.keys()]
if __name__ == "__main__":
    curr_dirr = Path(__file__).__str__().split('/')[0]
    data_dir = Path(curr_dirr) / "data"

    data_paths = list((data_dir / "seqcls").glob("*"))
    data_paths += list((data_dir / "tokcls").glob("*"))

    for dpath in data_paths:
        name = get_name(dpath)
        print("Dataset", name)
        bdata = get_link_preprocessed_data(dpath)
        assert(get_blurb_size(bdata) == BLURB_datasets[name], "Issue with " + name)
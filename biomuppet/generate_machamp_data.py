import json
from pathlib import Path

from datasets import DatasetDict
from tqdm import tqdm

from biomuppet.classification import get_all_classification_datasets
from biomuppet.coreference_resolution import get_all_coref_datasets
from biomuppet.relation_extraction import get_all_re_datasets
from biomuppet.semantic_textual_similarity import get_all_sts_datasets
from biomuppet.named_entity_recognition import get_all_ner_datasets
from biomuppet.utils import DEBUG


def clean_text(text):
    return text.strip().replace("\t", " ").replace("\n", " ")


if __name__ == "__main__":
    re_datasets = get_all_re_datasets()
    coref_datasets = get_all_coref_datasets()
    classification_datasets = get_all_classification_datasets()
    sts_datasets = get_all_sts_datasets()
    ner_datasets = get_all_ner_datasets()

    config = {}

    out = Path("machamp/data/bigbio/")
    out.mkdir(exist_ok=True, parents=True)

    # Process datasets

    ## Process classification-type datasets
    ### Add to Machamp config
    for dataset in tqdm(re_datasets + coref_datasets + classification_datasets):
        config[dataset.meta.name] = {
            "train_data_path": str((out / dataset.name).with_suffix(".train")),
            "validation_data_path": str((out / dataset.name).with_suffix(".valid")),
            "sent_idxs": [0],
            "tasks": {
                dataset.meta.name: {
                    "column_idx": 1,
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
    
        ### Write train file
        with (out / dataset.meta.name).with_suffix(".train").open("w", encoding="utf8") as f:
            for example in dataset.data["train"]:
                text = clean_text(example["text"])
                if not text:
                    continue
    
                label = "|".join(sorted(example["labels"]))
                if not label.strip():
                    label = "None"
    
                f.write(text + "\t" + label + "\n")
    
        ### Write validation file
        with (out / dataset.meta.name).with_suffix(".valid").open("w", encoding="utf8") as f:
            for example in dataset.data["valid"]:
                text = example["text"].strip().replace("\t", " ").replace("\n", " ")
                if not text:
                    continue
                label = "|".join(sorted(example["labels"]))
                if not label.strip():
                    label = "None"
    
                f.write(text + "\t" + label + "\n")

    
    for dataset in tqdm(sts_datasets):
        # WRONG, none of them are classification, except mqp (0/1 dissimilar/similar)
        config[dataset.meta.name] = {
            "train_data_path": str((out / dataset.meta.name).with_suffix(".train")),
            "validation_data_path": str((out / dataset.meta.name).with_suffix(".valid")),
            "sent_idxs": [0,1],
            "tasks": {
                dataset.meta.name: {
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

        def flatten_labels(example):
            example['labels'] = example['labels'][0]
            return example

        dataset.data = dataset.data.map(flatten_labels)
        dataset.data["train"].to_csv((out / dataset.meta.name).with_suffix(".train"), sep="\t", index=None, header=None)
        dataset.data["valid"].to_csv((out / dataset.meta.name).with_suffix(".valid"), sep="\t", index=None, header=None)

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
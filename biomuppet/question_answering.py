import json
from pathlib import Path
from typing import List

import datasets
from bigbio.utils.constants import Tasks
from datasets import DatasetDict
from nltk import WordPunctTokenizer
from tqdm import tqdm

from biomuppet.classification import get_classification_meta
from biomuppet.utils import SingleDataset, get_all_dataloaders_for_task, DEBUG


def is_valid_qa(example) -> bool:
    """
    Return non-empty text
    :param example:
    :return:
    """
    if "text" in example.keys():
        text = example["text"]
        return len(text) >=1
    else:
        sequence = example["sequence"]
        return len(sequence) >=1


def qa_to_classification(example):
    """
    Function that converts yesno and multiple choice QA to MaChamp classification
    :param example:
    :param mask_entities:
    :return:
    """
    if example["type"][0] == "yesno":
        new_example = {"text": [""], "labels": [[""]]}
        context = example["context"][0].strip().replace("\t", " ").replace("\n", " ")
        question = example["question"][0].strip().replace("\t", " ").replace("\n", " ")
        composite_text = context + " " + "[SEP]" + " " + question
        labels = [example["answer"][0][0].strip().replace("\t", " ").replace("\n", " ")]
        new_example["text"].append(composite_text)
        new_example["labels"].append(labels)
        return new_example

    elif example["type"][0] == "multiple_choice":
        new_example = {"text": [""], "labels": [[""]]}
        context = example["context"][0].strip().replace("\t", " ").replace("\n", " ")
        question = example["question"][0].strip().replace("\t", " ").replace("\n", " ")
        answer = example["answer"][0][0].strip().replace("\t", " ").replace("\n", " ")
        for choice in example["choices"][0]:
            answer_prompt = "Answer : {}".format(choice)
            composite_text = context + " " + "[SEP]" + " " + question + " " + answer_prompt
            if answer.lower() == choice.lower():
                labels = ["yes"]
            else:
                labels = ["no"]
            new_example["text"].append(composite_text)
            new_example["labels"].append(labels)
        return new_example

    else:
        return



#TODO: transforming QA to multiple choice
def qa_to_sequence(example):
    """
    Function that converts MCQA to MaChamp sequence
    Currently only converts data that has answer exactly matches that in the prompt

    1. Assert type =="multiple choice"
    2. Format sequence: tokens of the words (separated by empty line between samples)
    3. Format labels: [] or [answer]

    """
    if example["type"][0] != "multiple_choice":
        return
    else:
        new_example = {"sequence": [[""]], "labels": [[""]]}
        tokenizer = WordPunctTokenizer()
        context_seq = tokenizer.tokenize(example["context"][0].lower().strip().replace("\t", " ").replace("\n", " "))
        question_seq = tokenizer.tokenize(example["question"][0].lower().strip().replace("\t", " ").replace("\n", " "))
        sequence = context_seq + question_seq

        # Look for exact matches firstly:
        ans =example['answer'][0][0].lower().strip().replace("\t", " ").replace("\n", " ")
        if ans in sequence:
            labels = [["answer"] if (ans == seq) else [""] for seq in sequence]
            new_example["sequence"].extend([[seq] for seq in sequence])
            new_example["labels"].extend(labels)
            # Add new line to differentiate examples
            new_example["sequence"].extend([["\n"]])
            new_example["labels"].extend([["\n"]])
            return new_example
        # Currently doesn't deal with "soft matching" cases
        # TODO: implement when multiple choice answers "soft match" in the form used in the prompt
        else:
            return new_example


def get_all_qa_datasets() -> List[SingleDataset]:
    """
    Function that transforms QA dataset to MaChamp format;
    Transforms ALL QA dataset with yesno and multiple choice --> MaChamp classification
    Transforms some QA datasest with multiple choice that has exact match between answer and the prompt --> MaChamp
    Sequence
    Currenctly pulls biomrc, pubmed_qa, sciq. biomrc, pubmed_qa has subset_ids
    :return: qa_datasets
    """
    qa_clf_datasets = []
    qa_seq_datasets = []
    ignored_qa_datasets = ["mediqa_qa","biology_how_why_corpus", "med_qa", "bioasq_task_b", "medhop"]
    ds_subset_id = {'biomrc':['biomrc_large_B'],
                    'pubmed_qa':['pubmed_qa_labeled_fold0']}

    for dataset_loader in tqdm(
            get_all_dataloaders_for_task(Tasks.QUESTION_ANSWERING),
            desc="Preparing QA datasets",
    ):

        dataset_name = Path(dataset_loader).with_suffix("").name

        if DEBUG and "sciq" not in dataset_name:
            continue

        if dataset_name in ignored_qa_datasets:
            continue

        module = datasets.load.dataset_module_factory(str(dataset_loader))
        builder_cls = datasets.load.import_main_class(module.module_path)
        bigbio_config_names = [subset_id +"_bigbio_qa" for subset_id in ds_subset_id[dataset_name]] \
            if dataset_name in ds_subset_id.keys() else [el.name for el in builder_cls.BUILDER_CONFIGS if 'bigbio_qa' in el.name]
        print()
        for bigbio_config_name in bigbio_config_names:

            try:
                dataset = datasets.load_dataset(
                    str(dataset_loader), name=bigbio_config_name
                )
            except ValueError as ve:
                print(f"Skipping {dataset_loader} because of {ve}")
                continue
            dataset_filtered = dataset.filter(lambda x: len(x["context"])>1)
            # convert all QA dataset to MaChamp classification and add as SingleDataset
            new_dataset = dataset_filtered.map(
                qa_to_classification,
                batched=True,
                batch_size=1,
                remove_columns=dataset["train"].column_names,
            )
            new_dataset = new_dataset.filter(is_valid_qa)
            #TODO: write solution for meta information
            meta = get_classification_meta(dataset=new_dataset, name=bigbio_config_name[:-10] +"_CLF")
            qa_clf_datasets.append(SingleDataset(new_dataset, meta=meta))

            # If the dataset is MCQA, create MaChamp sequence format
            # convert all QA dataset to MaChamp classification and add as SingleDataset
            mcqa_dataset = dataset_filtered.filter(lambda x: x["type"]=="multiple_choice")
            if mcqa_dataset.num_rows['train'] >0:
                new_dataset = mcqa_dataset.map(
                    qa_to_sequence,
                    batched=True,
                    batch_size=1,
                    remove_columns=mcqa_dataset["train"].column_names,
                )
                new_dataset = new_dataset.filter(is_valid_qa)
                # for split_name, split in new_dataset.items():
                #     new_dataset[split_name] = subsample_negative(split)
                meta = get_classification_meta(dataset=new_dataset, name=bigbio_config_name[:-10]+"_SEQ")
                qa_seq_datasets.append(SingleDataset(new_dataset, meta=meta))

    return qa_clf_datasets, qa_seq_datasets



if __name__ == '__main__':
    qa_clf_datasets, qa_seq_datasets = get_all_qa_datasets()

    config = {}
    out = Path("machamp/data/bigbio/qa")
    out.mkdir(exist_ok=True, parents=True)

    for dataset in tqdm(qa_clf_datasets, desc="Writing QA-CLF"):
        name = dataset.meta.name
        config[name] = {
            "train_data_path": str((out / name).with_suffix(".train")),
            "validation_data_path": str((out / name).with_suffix(".valid")),
            "sent_idxs": [0, 1],
            "tasks": {
                name: {
                    "column_idx": 2,
                    "task_type": "classification"
                }
            }
        }
        ### Generate validation split if not available
        if not "validation" in dataset.data:
            train_valid = dataset.data["train"].train_test_split(test_size=0.1)
            dataset.data = DatasetDict({
                "train": train_valid["train"],
                "validation": train_valid["test"],
            })

        ### Write train file
        with (out / name).with_suffix(".train").open("w") as f:
            for example in dataset.data["train"]:
                text = example["text"].strip().replace("\t", " ").replace("\n", " ") if "text" in example.keys() \
                    else example["sequence"][0]
                text = text.replace("[SEP]", "\t")
                if not text:
                    continue
                label = "|".join(sorted(example["labels"]))
                if not label.strip():
                    label = "None"

                f.write(text + "\t" + label + "\n")

        ### Write validation file
        with (out / name).with_suffix(".valid").open("w") as f:
            for example in dataset.data["validation"]:
                text = example["text"].strip().replace("\t", " ").replace("\n", " ") if "text" in example.keys() \
                    else example["sequence"][0]
                text = text.replace("[SEP]", "\t")
                if not text:
                    continue
                label = "|".join(sorted(example["labels"]))
                if not label.strip():
                    label = "None"

                f.write(text + "\t" + label + "\n")


    for dataset in tqdm(qa_seq_datasets, desc="Writing QA-SEQ"):
        name = dataset.meta.name
        config[name] = {
            "train_data_path": str((out / name).with_suffix(".train")),
            "validation_data_path": str((out / name).with_suffix(".valid")),
            "word_idxs": 0,
            "tasks": {
                name: {
                    "column_idx": 1,
                    "task_type": "sequence"
                }
            }
        }
        ### Generate validation split if not available
        if not "validation" in dataset.data:
            train_valid = dataset.data["train"].train_test_split(test_size=0.1)
            dataset.data = DatasetDict({
                "train": train_valid["train"],
                "validation": train_valid["test"],
            })

        ### Write train file
        with (out / name).with_suffix(".train").open("w") as f:
            for example in dataset.data["train"]:
                text = example["text"].strip().replace("\t", " ").replace("\n", " ") if "text" in example.keys() \
                    else example["sequence"][0]
                if not text:
                    continue
                label = "|".join(sorted(example["labels"]))
                if not label.strip():
                    label = "None"

                f.write(text + "\t" + label + "\n")
            f.write("\n")

        ### Write validation file
        with (out / name).with_suffix(".valid").open("w") as f:
            for example in dataset.data["validation"]:
                text = example["text"].strip().replace("\t", " ").replace("\n", " ") if "text" in example.keys() \
                    else example["sequence"][0]
                if not text:
                    continue
                label = "|".join(sorted(example["labels"]))
                if not label.strip():
                    label = "None"

                f.write(text + "\t" + label + "\n")
            f.write("\n")

    with (out / "config.json").open("w") as f:
        json.dump(config, f)
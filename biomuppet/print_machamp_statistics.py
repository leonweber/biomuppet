import json
import argparse
from tqdm import tqdm
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    args.config = "/vol/fob-wbib-vol2/wbi/weberple/projects/biomuppet/machamp/configs/bigbio_full.json"
    with open(args.config) as f:
        config = json.load(f)

    df = {"task_type": [], "task": [], "num_train_examples": [], "num_validation_examples": []}
    for task, task_config in tqdm(list(config.items())):
        task_type = task.split("_")[0]
        if "edge" in task:
            task_type = "EAE"
        machamp_task_type = list(task_config["tasks"].values())[0]["task_type"]
        num_examples = 0
        with open("machamp/" + task_config["train_data_path"]) as f:
            if machamp_task_type == "seq":
                for line in f:
                    if line.strip() == "":
                        num_examples += 1
            elif machamp_task_type == "classification":
                for line in f:
                    num_examples += 1
            else:
                raise ValueError("Unknown task type: {}".format(machamp_task_type))
        df["num_train_examples"].append(num_examples)

        with open("machamp/" + task_config["validation_data_path"]) as f:
            num_examples = 0
            if machamp_task_type == "seq":
                for line in f:
                    if line.strip() == "":
                        num_examples += 1
            elif machamp_task_type == "classification":
                for line in f:
                    num_examples += 1
            else:
                raise ValueError("Unknown task type: {}".format(machamp_task_type))
            df["num_validation_examples"].append(num_examples)
        
        df["task_type"].append(task_type)
        df["task"].append(task)
    df = pd.DataFrame(df)
    print(df.groupby("task_type").sum())
    print(df["task_type"].value_counts())
    
            

    
import json
from pathlib import Path

from biomuppet.utils import DEBUG

if __name__ == '__main__':
    full_config = {}
    for config_file in Path("machamp/data/bigbio").glob("*/config.json"):
        with open(config_file) as f:
            config = json.load(f)
            for i, (task_name, task_data) in enumerate(config.items()):
                if DEBUG and i >= 3:
                    break
                task_data = task_data.copy()
                task_name = config_file.parent.name + "_" + "_".join(task_name.split("_")[:-1])
                task_data["train_data_path"] = task_data["train_data_path"].replace("machamp/", "")
                task_data["validation_data_path"] = task_data["validation_data_path"].replace("machamp/", "")

                full_config[task_name] = task_data


    if DEBUG:
        with open("machamp/configs/bigbio_debug.json", "w") as f:
            json.dump(full_config, f, indent=1)
    else:
        with open("machamp/configs/bigbio_full.json", "w") as f:
            json.dump(full_config, f, indent=1)

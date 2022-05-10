import json
from glob import glob

if __name__ == '__main__':
    full_config = {}
    for config_file in glob("machamp/data/bigbio/*/config.json"):
        with open(config_file) as f:
            config = json.load(f)
            for task in config.values():
                task["train_data_path"] = task["train_data_path"].replace("machamp/", "")
                task["validation_data_path"] = task["validation_data_path"].replace("machamp/", "")

            full_config.update(config)

    with open("machamp/configs/bigbio_full.json", "w") as f:
        json.dump(full_config, f, indent=1)

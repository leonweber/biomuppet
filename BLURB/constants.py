# Task mapping

task_mapping = {
    "classification": "_text",
    "coref": "_coref",
    "event_extraction": "_RE",
    "named_entity_recognition": "_ner",
    "qa": ["_CLF", "_SEQ"],
    "re": "_RE",
    "sts": "_sts",
    "event_extraction_trigger_recognition": "_NER",
    "event_extraction_edge_classification": "_RE",
}

# Text pairs

text_pairs = [
    "BIOSSES_hf",
    "pubmedqa_hf",
    "bioasq_hf",
]

ner_examples = [
    "BC5CDR-disease_hf",
    "JNLPBA_hf",
    "BC5CDR-chem_hf",
    "NCBI-disease_hf",
    "ebmnlp_hf",
    "BC2GM_hf",
]

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
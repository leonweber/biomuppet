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

# PUBMED QA is the only big one; process this separately
bb_configs = {
    "bc5cdr": "bc5cdr_bigbio_kb",
    "gnormplus": "gnormplus_bigbio_kb",
    "ncbi_disease": "ncbi_disease_bigbio_kb",
    "ebm_pico": "ebm_pico_bigbio_kb",
    "chemprot": "chemprot_bigbio_kb", # "chemprot_shared_task_eval_source", 
    "ddi_corpus": "ddi_corpus_bigbio_kb",
    "biosses": "biosses_bigbio_pairs",
    "hallmarks_of_cancer": "hallmarks_of_cancer_bigbio_text",
    "pubmed_qa": ["pubmed_qa_labeled_fold" + str(i) for i in range(1, 11)],
    "bioasq_task_b": "bioasq_7b_bigbio_qa",
}
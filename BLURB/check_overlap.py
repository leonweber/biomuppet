import gzip
import pickle as pkl


if __name__ == "__main__":

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

# Full set across ALL TASKS
full_machamp_train = set().union(*list(machamp_train.values()))
full_machamp_val = set().union(*list(machamp_val.values()))

# Compare overlap between all <train> ---- <train>
print(
    "BLURB Train v. MACHAMP Train\n",
    len(blurb["train"].intersection(full_machamp_train)),
    "/", len(blurb["train"]), "\n",
    len(blurb_ner["train"].intersection(full_machamp_train)),
    "/", len(blurb_ner["train"]), " (NER) \n",
    len(blurb_text_pairs["train"].intersection(full_machamp_train)),
    "/", len(blurb_text_pairs["train"]), " (text pairs)\n",
)

# Compare overlap between all <train> ---- <val>

print(
    "BLURB Train v. MACHAMP Val\n",
    len(blurb["train"].intersection(full_machamp_val)),
    "/", len(blurb["train"]), "\n",
    len(blurb_ner["train"].intersection(full_machamp_val)),
    "/", len(blurb_ner["train"]), " (NER) \n",
    len(blurb_text_pairs["train"].intersection(full_machamp_val)),
    "/", len(blurb_text_pairs["train"]), " (text pairs)\n",
)


# Compare overlap between all <dev> ---- <train>

print(
    "BLURB Dev v. MACHAMP Train\n",
    len(blurb["dev"].intersection(full_machamp_train)),
    "/", len(blurb["dev"]), "\n",
    len(blurb_ner["dev"].intersection(full_machamp_train)),
    "/", len(blurb_ner["dev"]), " (NER) \n",
    len(blurb_text_pairs["dev"].intersection(full_machamp_train)),
    "/", len(blurb_text_pairs["dev"]), " (text pairs)\n",
)


# Compare overlap between all <dev> ---- <val>

print(
    "BLURB Dev v. MACHAMP Val\n",
    len(blurb["dev"].intersection(full_machamp_val)),
    "/", len(blurb["dev"]), "\n",
    len(blurb_ner["dev"].intersection(full_machamp_val)),
    "/", len(blurb_ner["dev"]), " (NER) \n",
    len(blurb_text_pairs["dev"].intersection(full_machamp_val)),
    "/", len(blurb_text_pairs["dev"]), " (text pairs)\n",
)

# Compare overlap between all <test> ---- <train>

print(
    "BLURB test v. MACHAMP train\n",
    len(blurb["test"].intersection(full_machamp_train)),
    "/", len(blurb["test"]), "\n",
    len(blurb_ner["test"].intersection(full_machamp_train)),
    "/", len(blurb_ner["test"]), " (NER) \n",
    len(blurb_text_pairs["test"].intersection(full_machamp_train)),
    "/", len(blurb_text_pairs["test"]), " (text pairs)\n",
)

# Compare overlap between all <test> ---- <val>

print(
    "BLURB test v. MACHAMP Val\n",
    len(blurb["test"].intersection(full_machamp_val)),
    "/", len(blurb["test"]), "\n",
    len(blurb_ner["test"].intersection(full_machamp_val)),
    "/", len(blurb_ner["test"]), " (NER) \n",
    len(blurb_text_pairs["test"].intersection(full_machamp_val)),
    "/", len(blurb_text_pairs["test"]), " (text pairs)\n",
)
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


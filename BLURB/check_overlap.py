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


 #  ------------------------------------------------ #
    for dset in machamp_train:

        print(
            "BLURB Train v. Machamp Train; " + dset  + "\n",
            len(blurb["train"].intersection(machamp_train[dset])),
            "/", len(blurb["train"]), "\n",
        )
        print(
            "BLURB NER sentences v. Machamp Train; " + dset  + "\n",
            len(blurb_ner["train"].intersection(machamp_train[dset])),
            "/", len(blurb_ner["train"]), " (NER) \n",
        )
        print(
            "BLURB Text Pairs v. Machamp Train; " + dset  + "\n",
            len(blurb_text_pairs["train"].intersection(machamp_train[dset])),
            "/", len(blurb_text_pairs["train"]), " (text pairs) \n",
        )
        print("-------")

    print("\n\n=====================")

    for dset in machamp_train:

        print(
            "BLURB test v. Machamp Train; " + dset  + "\n",
            len(blurb["test"].intersection(machamp_train[dset])),
            "/", len(blurb["test"]), " \n",
        )
        print(
            "BLURB NER sentences test v. Machamp Train; " + dset  + "\n",
            len(blurb_ner["test"].intersection(machamp_train[dset])),
            "/", len(blurb_ner["test"]), " (NER) \n",
        )
        print(
            "BLURB Text Pairs test v. Machamp Train; " + dset  + "\n",
            len(blurb_text_pairs["test"].intersection(machamp_train[dset])),
            "/", len(blurb_text_pairs["test"]), "  (text pairs) \n",
        )
        print("-------")

    print("\n\n=====================")

    for dset in machamp_train:

        print(
            "BLURB dev v. Machamp Train; " + dset  + "\n",
            len(blurb["dev"].intersection(machamp_train[dset])),
            "/", len(blurb["dev"]), " \n",
        )
        print(
            "BLURB NER sentences dev v. Machamp Train; " + dset  + "\n",
            len(blurb_ner["dev"].intersection(machamp_train[dset])),
            "/", len(blurb_ner["dev"]), " (NER) \n",
        )
        print(
            "BLURB Text Pairs dev v. Machamp Train; " + dset  + "\n",
            len(blurb_text_pairs["dev"].intersection(machamp_train[dset])),
            "/", len(blurb_text_pairs["dev"]), "  (text pairs) \n",
        )
        print("-------")
 #  ------------------------------------------------ #


    for dset in machamp_val:

        print(
            "BLURB Train v. Machamp Val; " + dset  + "\n",
            len(blurb["train"].intersection(machamp_val[dset])),
            "/", len(blurb["train"]), " \n",
        )
        print(
            "BLURB NER sentences v. Machamp Val; " + dset  + "\n",
            len(blurb_ner["train"].intersection(machamp_val[dset])),
            "/", len(blurb_ner["train"]), " (NER) \n",
        )
        print(
            "BLURB Text Pairs v. Machamp Val; " + dset  + "\n",
            len(blurb_text_pairs["train"].intersection(machamp_val[dset])),
            "/", len(blurb_text_pairs["train"]), " (text pairs)  \n",
        )
        print("-------")

    print("\n\n=====================")

    for dset in machamp_val:

        print(
            "BLURB test v. Machamp Val; " + dset  + "\n",
            len(blurb["test"].intersection(machamp_val[dset])),
            "/", len(blurb["test"]), "\n",
        )
        print(
            "BLURB NER sentences test v. Machamp Val; " + dset  + "\n",
            len(blurb_ner["test"].intersection(machamp_val[dset])),
            "/", len(blurb_ner["test"]), " (NER) \n",
        )
        print(
            "BLURB Text Pairs test v. Machamp Val; " + dset  + "\n",
            len(blurb_text_pairs["test"].intersection(machamp_val[dset])),
            "/", len(blurb_text_pairs["test"]), " (text pairs) \n",
        )
        print("-------")

    print("\n\n=====================")
    
    for dset in machamp_val:

        print(
            "BLURB dev v. Machamp Val; " + dset  + "\n",
            len(blurb["dev"].intersection(machamp_val[dset])),
            "/", len(blurb["dev"]), " \n",
        )
        print(
            "BLURB NER sentences dev v. Machamp Val; " + dset  + "\n",
            len(blurb_ner["dev"].intersection(machamp_val[dset])),
            "/", len(blurb_ner["dev"]), " (NER) \n",
        )
        print(
            "BLURB Text Pairs dev v. Machamp Val; " + dset  + "\n",
            len(blurb_text_pairs["dev"].intersection(machamp_val[dset])),
            "/", len(blurb_text_pairs["dev"]), " (text pairs) \n",
        )
        print("-------")
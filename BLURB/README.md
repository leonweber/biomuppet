# Quantifying overlap of train/test BLURB datasets

### 2022.05.17

Considering low overlaps, we will modify the problem.

Each BLURB and MACHAMP dataset has sets of sentences to compare. We will do the following:
    - Every 'sentence' element is split into a set of unique tokens
    - All tokens are string normalized (cleaned, white-space stripped).
    - Punctuations + stop words omitted.

We consider overlaps in the examples if some 80% of tokens between a BLURB bag-of-words and MACHAMP bag-of-words are the same

### 2022.05.16

Retrieval against a set of sentences; if the best match is above some threshold, we count it as a perfect match
Who is doing this??

To do - make each items a bag of words retrieval baseline.
Save the intersection of the dataset

### 2022.05.14

Updated to reflect changes:
    - joins tokens in NER task
    - "cleans" strings for excessive white space using `named_entity_recognition` helper functions
    - Returns MACHAMP data tokens per dataset


Some notes on parsing the MACHAMP data:
Empty dataset with dataset /home/natasha/Projects/biomuppet/machamp/data/bigbio/qa/biomrc_large_B_SEQ.train
Empty dataset with dataset /home/natasha/Projects/biomuppet/machamp/data/bigbio/qa/biomrc_large_B_SEQ.valid

### 2022.05.11
Disregard NER - this was done via token-specific overlap; we needed to join the tokens together

Output of overlap (WITHOUT NER)
BLURB Train v. MACHAMP Train
 8009 / 60212 
 19114 / 76802  (NER) 
 57 / 2358  (text pairs)

BLURB Train v. MACHAMP Val
 898 / 60212 
 16651 / 76802  (NER) 
 6 / 2358  (text pairs)

BLURB Dev v. MACHAMP Train
 604 / 16195 
 12966 / 32043  (NER) 
 17 / 282  (text pairs)

BLURB Dev v. MACHAMP Val
 71 / 16195 
 11786 / 32043  (NER) 
 0 / 282  (text pairs)

BLURB test v. MACHAMP Val
 2 / 25716 
 12898 / 30096  (NER) 
 14 / 1319  (text pairs)

BLURB test v. MACHAMP Val
 0 / 25716 
 11896 / 30096  (NER) 
 4 / 1319  (text pairs)

### INSTRUCTIONS TODO
First, run the script `fetch_blurb.py`.

This may require different system requirements. The only noticeable issue with `bigbio` and is the `bioc` requirement needs a different vesion of `lxml` (a higher version, noticeably)

Colab notebook: https://worksheets.codalab.org/worksheets/0x7a6ab9c8d06a41d191335b270da2902e

BLURB Table: https://microsoft.github.io/BLURB/tasks.html

PubMedQA is probably pqal

NOTE, processing via the actual BLURB dataset will require python < 3.9 as the XML library works differently

original blurb: https://microsoft.github.io/BLURB/sample_code/data_generation.tar.gz

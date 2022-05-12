# Quantifying overlap of train/test BLURB datasets

# 2022.05.11

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

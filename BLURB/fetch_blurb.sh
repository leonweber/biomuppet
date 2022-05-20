#!/usr/bin/env bash

# Fetches BLURB dataset PREPROCESSED via LINKBERT
wget https://nlp.stanford.edu/projects/myasu/LinkBERT/data.zip
unzip data.zip

# Generate MACHAMP data, if not already provided
python -m biomuppet.classification
python -m biomuppet.relation_extraction
python -m biomuppet.named_entity_recognition  # this fails
python -m biomuppet.event_extraction
python -m biomuppet.coreference_resolution
python -m biomuppet.question_answering  # this hangs
python -m biomuppet.semantic_textual_similarity

python get_overlap.py
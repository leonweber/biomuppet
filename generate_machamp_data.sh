#!/usr/bin/env bash

if [ $# -eq 0 ]
then
  Usage: bash generate_machamp_data.sh [MACHAMP_ROOT_DIR]
  exit
fi

#machamp_dir="$1"
#rm -rf "$machamp_dir"/params
#ln -s "$(pwd)/machamp/data" "$machamp_dir"
#ln -s "$(pwd)/machamp/configs" "$machamp_dir"

python -m biomuppet.classification
python -m biomuppet.coreference_resolution
python -m biomuppet.event_extraction
python -m biomuppet.named_entity_recognition
python -m biomuppet.question_answering
python -m biomuppet.relation_extraction
python -m biomuppet.semantic_textual_similarity

python -m biomuppet.generate_config

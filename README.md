# biomuppet

Code for training a massive multitask learning model (like [MUPPET](https://arxiv.org/abs/2101.11038?utm_campaign=NLP%20News&utm_medium=email&utm_source=Revue%20newsletter)) on the bigbio data.

## How to train
For now, the script assumes that it resides in the bigbio root (`biomedical/`). 
This is also why everything is in a single file. As soon as bigbio is installable as a library, we should refactor this into multiple files.

To train, simply call the script: `python train_biomuppet.py`

## Currently supported tasks
- **Relation Extraction**: Transform into text classification by using the relation classification approach. 
  E.g. `A is related to B and not to C` would be translated into six examples `[HEAD-S]A[HEAD-E] is related to [TAIL-S]B[TAIL-E] and not to C`, `[HEAD-S]A[HEAD-E] is related to B and not to [TAIL-S]C[TAIL-E]` etc.
  and then each would be classified seperately. Negative examples (without any relation) are subsampled to not exceed 10x the number of positive examples
- **Text classification**: Text classification examples are taken as-is.

## TODOs
- Implement further tasks. Maybe with high priority those that are contained in the BLURB Benchmark (or whatever we choose for eval)
  * **Named Entity Recognition (High priority)**: Can be either modeled as sequence labelling or as span classification.
  * **Question Answering (High priority)**
  * **Coreference Resolution**: I'd propose to model this like Relation Extraction with a single Relation (is-coref-of)
  * **Event trigger detection**: Can be modelled exactly like NER
  * **Textual entailment**
- Implement loss normalization so that all loss values live on the same scale (see MUPPET paper)
- Decide on evaluation benchmark and implement it (maybe use BLURB?)

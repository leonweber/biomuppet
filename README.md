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

## How to add further tasks
1. Implement `get_all_{task}_datasets(tokenizer, split="train")`. Should return `datasets.Dataset` with (a) columns produced by the tokenizer (`input_ids`, `attention_mask`, `token_type_ids` (for BERT-based models)) and (b) a `labels` column that contains one list/tensor of labels per example, which will be used by the loss function.
2. Implement or reuse existing `get_{task}_meta(dataset, name)` which should return an instance of `DatasetMetaInformation` that provides the necessary information about the task to the loss function.
3. Implement or reuse existing `{task}_loss(logits, labels, meta)`. `logits` is a tensor of shape `batch_size, seq_len, out_dim` with `out_dim == len(meta.label_to_id)`. `labels` are the contents of the `labels` column defined in step 1. `meta` is the task meta information returned by the function defined in step 2.

For an example implementation, please take a look at the already existing tasks `relation_extraction` and `classification`.

## TODOs
- Implement further tasks. Maybe with high priority those that are contained in the BLURB Benchmark (or whatever we choose for eval)
  * **Named Entity Recognition (High priority)**: Can be either modeled as sequence labelling or as span classification.
  * **Question Answering (High priority)**
  * **Coreference Resolution**: I'd propose to model this like Relation Extraction with a single Relation (is-coref-of)
  * **Event trigger detection**: Can be modelled exactly like NER
  * **Textual entailment**
- Implement loss normalization so that all loss values live on the same scale (see MUPPET paper)
- Decide on evaluation benchmark and implement it (maybe use BLURB?)

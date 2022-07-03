#%%
import argparse
from pathlib import Path

import torch
import transformers

#%%



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--machamp_model", required=True, type=Path)
    parser.add_argument("--transformer", default="michiyasunaga/BioLinkBERT-base")
    parser.add_argument("--hub_name", required=True)
    args = parser.parse_args()

    weights = torch.load(args.machamp_model)
    transformer_state_dict = {k.replace("_text_field_embedder.token_embedder_tokens._matched_embedder.transformer_model.", "bert."): v for k, v in weights.items() if "transformer" in k}
#%%
    id2label = {}
    hf_weights = []
    hf_biases = []
    tasks = sorted(set(k.split(".")[1] for k in weights if "decoders" in k))
    for task in tasks:
        vocab_file = args.machamp_model.parent / "vocabulary" / f"{task}.txt"
        if not vocab_file.exists():
            continue
        with open(vocab_file) as f:
            vocab = [line.strip() for line in f]
            task_weights = next(v for k,v in weights.items() if "decoders" in k and task in k and "weight" in k)
            task_bias = next(v for k,v in weights.items() if "decoders" in k and task in k and "bias" in k)
            for label in vocab[1:]: # skip @@UNKNOWN@@
                id2label[len(id2label)] = f"{task}:{label})"
            hf_weights.append(task_weights[2:])
            hf_biases.append(task_bias[2:])
    hf_weights = torch.cat(hf_weights, dim=0)
    hf_biases = torch.cat(hf_biases, dim=0)
    transformer_state_dict['classifier.weight'] = hf_weights
    transformer_state_dict['classifier.bias'] = hf_biases
    transformer_state_dict.pop("bert.pooler.dense.weight")
    transformer_state_dict.pop("bert.pooler.dense.bias")


#%% 
    model = transformers.AutoModelForTokenClassification.from_pretrained(
        args.transformer,
        num_labels=len(id2label),
        id2label=id2label,
        )
    
    model.load_state_dict(transformer_state_dict)
    model.push_to_hub(args.hub_name)
    

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.transformer)
    tokenizer.push_to_hub(args.hub_name)

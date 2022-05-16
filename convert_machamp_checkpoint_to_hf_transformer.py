import argparse
from pathlib import Path

import torch
import tarfile
import transformers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--machamp_model", required=True, type=Path)
    parser.add_argument("--transformer", default="michiyasunaga/BioLinkBERT-base")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    weights = torch.load(args.machamp_model)
    transformer_state_dict = {k.replace("_text_field_embedder.token_embedder_tokens._matched_embedder.transformer_model.", ""): v for k, v in weights.items() if "transformer" in k}

    model = transformers.AutoModel.from_pretrained(args.transformer)
    model.load_state_dict(transformer_state_dict)



    model.save_pretrained(args.out)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.transformer)
    tokenizer.save_pretrained(args.out)
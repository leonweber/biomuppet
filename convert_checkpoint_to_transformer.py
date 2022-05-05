import argparse
from pathlib import Path

from train_biomuppet import BioMuppet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    model = BioMuppet.load_from_checkpoint(
        args.checkpoint,
        transformer="michiyasunaga/BioLinkBERT-base",
        lr=3e-5,
        dataset_to_meta={},
        strict=False
    )
    breakpoint()

    model.transformer.save_pretrained(args.out)
    model.tokenizer.save_pretrained(args.out)



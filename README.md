### Setup
1. Install `bigbio`
```
git clone git@github.com:bigscience-workshop/biomedical.git
cd biomedical
pip install -e .
cd ..
```
2. Install our fork of `allennlp v 1.3` with [DDP + gradient accumulation patch](https://github.com/allenai/allennlp/pull/5100) backported
```
pip install git+https://github.com/leonweber/allennlp
```
3. Uninstall the installed CPU pytorch and install GPU pytorch
```
pip uninstall pytorch
pip uninstall torch
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
``` 
5. Clone our fork of [Machamp](https://github.com/machamp-nlp/machamp):
```
git clone git@github.com:leonweber/machamp.git
```
6. Clone the biomuppet repository and install dependencies
```
git clone git@github.com:leonweber/biomuppet.git
pip install -r biomuppet/requirements.txt
```
7. Generate the machamp training data
```
cd biomuppet; bash generate_machamp_data.sh [MACHAMP_ROOT_PATH]; cd ..
```
8a. Run Machamp training (single node)
```
cd [MACHAMP_ROOT_PATH]; python train.py --dataset_config configs/bigbio_debug.json
```
8b. Run Machamp training (multiple nodes)

Set correct distributed settings in `[MACHAMP_ROOT_PATH]/configs/params.json
```
    "distributed": {
        "cuda_devices": [0, 1, 2, 3], # note that all nodes have to have the same number of GPUs for AllenNLP multi node training to work
        "master_address": "[ADDRESS of main node]",
        "master_port": "29500", # some free port
        "num_nodes": [Total number of nodes]
```

Start training
```
cd [MACHAMP_ROOT_PATH]; python train.py --dataset_config configs/bigbio_debug.json --node_rank [rank of local node]
```

### Project Documentation
For the project description and open todos, see the project doc on:
https://docs.google.com/document/d/11w_XxFMrMnRD_FX1oi-tGWI5Z3k7QXB6s7H-GuIr2ow/edit?usp=sharing



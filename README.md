### Setup
1. Install `bigbio`
```
pip install git+https://github.com/bigscience-workshop/biomedical
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
6. Clone the biomuppet repository
```
git clone git@github.com:leonweber/biomuppet.git
```
7. Generate the machamp training data
```
cd biomuppet; bash generate_machamp_data.sh [MACHAMP_ROOT_PATH]
```
8. Run Machamp training
```
cd [MACHAMP_ROOT_PATH]; python train.py --dataset_config configs/bigbio_debug
```

### Project Documentation
For the project description and open todos, see the project doc on:
https://docs.google.com/document/d/11w_XxFMrMnRD_FX1oi-tGWI5Z3k7QXB6s7H-GuIr2ow/edit?usp=sharing



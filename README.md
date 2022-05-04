### Setup
This project requires `bigbio` package as the dependency. To install `bigbio` package, we can run:
```
git clone git@github.com:bigscience-workshop/biomedical.git
cd biomedical
pip install -e ".[dev]"
```
We can verify the installation is correct by executing `from bigbio.utils.constants import Tasks` in our python script.

Since the existing data path is hardcoded, we also need to link the `biodatasets` folder from the [`biomedical`](https://github.com/bigscience-workshop/biomedical) repository. We can achieve that through by linking the folder with:
```
ln -s <PATH_TO_BIOMEDICAL_DIRECTORY>/biodatasets ./biodatasets
```

After setting the dependency, you can simply running the training code with:
```
python train_biomuppet.py --output_dir <PATH_TO_RESULT_DIRECTORY>
```

### Project Documentation
For the project description and open todos, see the project doc on:
https://docs.google.com/document/d/11w_XxFMrMnRD_FX1oi-tGWI5Z3k7QXB6s7H-GuIr2ow/edit?usp=sharing



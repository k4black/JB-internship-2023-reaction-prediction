# JB-int-2023


JetBrαins interηship 2023 test assignment repository. The "Predictioη of reαctions to the posts" problem.
(the special symbols added to remove repo from GitHub search)

The report is available in `report.pdf` file. The experiments tracking data is available in [neptune.ai project](https://new-ui.neptune.ai/k4black/
jb-reaction-prediction).


## Structure 

The following files are provided:
* `dataset.csv` - provided dataset with labels changed to the `negative`, `neutral`, `positive`;
* `scripts/`:
  * `eda.py` - simple script to got dataset statistics;
  * `classical_ml.py` - script evaluate 7 classical ML models against 4 vectorizers;
  * `finetune.py` - script to finetune any HuggingFace-based model using `params.json` parameters (cross-validation is available);
  * `efl.py` - script to finetune model using "Entailment as Few-Shot Learner" (EFL) method.
  * `params.json` - set of parameters for finetuning the DL models. The param set fall back to its prefix (e.g. `default-b8` will extend `default`);
* `colab-run.ipynb` - notebook to run the scripts on colab;
* `report.pdf` - Experiments report.

For the parameters see `python <script_name>.py --help` option.


## Requirements

Required: Python version 3.9 as colab have this version at the moment. 

For the local usage venv usage is recommended:
```shell
python -m venv .venv
source .venv/bin/activate 
python -m pip install -r requirements.txt
```

For machine learning: 
* FastText vectors are used: [cc.en.300.bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz) have to be downloaded;
* As well as the spacy model: `python -m spacy download en_core_web_sm`

For deep learning: 
* The project uses neptune.ai for experiments tracking. So, the training scripts require `NEPTUNE_PROJECT` and `NEPTUNE_API_TOKEN` environment variables to be set.


## Usage 

* For Local usage:   
  Each `.py` file provided is standalone script required only `dataset.csv` and `params.json` file to operate.  
  It can be run with `python <file_name>.py` command. For the parameters see `--help` option.

* For Colab usage: see `colab-run.ipynb` notebook.  
  This notebook only runs the specified files to use colab gpu.   
  The required `*.py` files and `params.json` have to be copied in the `content/` folder.

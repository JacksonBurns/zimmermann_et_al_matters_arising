# Comment on _A chemical language model for molecular taste prediction_

This repo contains the code supporting the brief article "Comment on _A chemical language model for molecular taste prediction_".
The original article can be found at DOI [10.1038/s41538-025-00474-z](https://doi.org/10.1038/s41538-025-00474-z).

This code has been adapted from the source code provided on [GitHub](https://github.com/fart-lab/fart).
Model definitions and hyperparameters, in particular, are re-used exactly as specified in the original paper.

Python 3.12 was used for the development of this code, though Python 3.11 should also work.
To run the code shown here, you must install the following packages:
```
statsmodels
datasets
transformers[torch]
numpy
scikit-learn
evaluate
torch
rdkit
seaborn
matplotlib
chemprop
xgboost
pandas
imbalanced-learn
chemprop
mordredcommunity
scikit-mol
tabulate
```

The exact versions used in creating this code are given in [`requirements.txt`](./requirements.txt).
They may not work on all platforms, in which case the above list of requirements should be sufficient.

`sckit-learn` compatible wrapper for Chemprop and ChemBERTa are provided in their respective `*_estimator.py` files.
Other tested models are natively compatible with `sklearn`.
To generate the results from scratch, first run `RUN_ZIMMERMANN=1 accelerate launch fit.py` and then after that is finished run `python fit.py`.
This script must be run twice since Chemprop and ChemBERTa use different, incompatible training backends.
Previous models will not be re-run, since results are saved to disk between executions (`outputs_*`).

Data curation and results are in the `data.ipynb` file and `data` directory, respectively.

Comment paper content is in the `paper` directory.

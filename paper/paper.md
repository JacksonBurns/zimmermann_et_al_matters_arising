---
title: "Matters Arising: _A chemical language model for molecular taste prediction_"
author:
- name: Jackson W. Burns
  id: jwb
  equal_contributor: yes
  institute:
  - 1
- name: Jonathan W. Zheng
  id: jwz
  equal_contributor: yes
  institute:
  - 1
- name: William H. Green
  id: whg
  correspondence: yes
  email: 'whgreen\@mit.edu'
  institute:
  - 1
institute:
- id: mit
  name: Massachusetts Institute of Technology
date: 12 September 2025
geometry: margin=1in
bibliography: paper.bib
citation-style: nature
colorlinks: true
note: |
 This paper can be compiled to other formats (pdf, word, etc.) using pandoc:
   pandoc --lua-filter=author-info-blocks.lua -H disable_float.tex --citeproc -s paper.md -o paper.docx
 from the paper directory (you can also export a pdf by using paper.pdf as the output).
---

# Abstract

Zimmermann _et al._[@zimmermann] collated a dataset of ~15,000 molecules with taste labels and trained a machine learning model which they claimed outperformed baselines.
We identified data issues including train/test leakage, improper curation, and incorrect label choices.
Additional methodological concerns include missing statistical tests and erroneous handling of data augmentation during evaluation.
Our analysis reveals that simple baselines outperform their models, disagreeing with the conclusions of Zimmerman _et al._

# Background

Machine learning has emerged as a revolutionary technology across the chemical sciences, ranging in applications from drug discovery to materials design.
We were pleased to see the extension of machine learning to food science in the work of Zimmermann _et al._ [@zimmermann]

Our prior experience in chemical machine learning has exposed us to common pitfalls, which can lead to overestimated model performance.
We identified some of those same issues in Zimmermann _et al._ in regard to data curation and evaluation of model performance.
We hope that the suggestions provided herein will help support the continued advancement of machine learning in food science.

## Model and Dataset Overview

Flavor Analysis and Recognition Transformer (FART) is a transformer-based modeling approach[@zimmermann] derived from the ChemBERTa[@chemberta] model via fine-tuning on the FartDB dataset.
FartDB is composed of other datasets: ChemTastesDB,[@rojas2022chemtastesdb], FlavorDB,[@garg2018flavordb], PlantMolecularTasteDB,[@gradinaru2022plantmoleculartastedb], TAS2R Agonists (bitter molecules),[@bayer2021chemoinformatics], IUPAC Dissociation Constants [@iupac_pka_database; @perrin1965dissociation; @perrin1972dissociation; @serjeant1979ionisation] (acids between 2-7 $\mathrm{p}K_{\mathrm{a}}$ were labeled as sour), and Suess et al. [@suess2015umami] umami compounds.

## Model Training

In the reference study FartDB was randomly partitioned into training, validation, and testing sets.
The following models were then trained and optimized using the training and validation sets: `FART` (here called `zimmermann` to distinguish between the _model_ and the _architecture_), `FART augmented` (`zimmermann-augmented`), `FART augmented + confidence` (`zimmermann-augmented-confidence`), `xgb-fingerprint-mordred`, `xgb-fingerprint`, `chemprop`, and `balancedforest`.

For the reported `zimmermann-augmented` model, the authors created approximately 10 unique SMILES strings for each training compound, reusing the same flavor label for each.
This established approach helps mitigate the impact of SMILES' non-deterministic semantics on model learning.
This same augmentation procedure was conducted on the testing data, yielding multiple predictions for each input.

While useful to end users, the `zimmermann-augmented-confidence` model is excluded from this analysis.
This model's 'all-or-nothing' voting choice (`confidence`) scheme for deriving a prediction from the augmented outputs allows the model to refuse to make a prediction.
To facilitate fair comparisons, a refused prediction is a wrong prediction when there exist models capable of at least attempting a prediction; the reported metrics should reflect this, rather than being reframed as "support" of the test set as done in the reference study.

To arrive at their final models, the authors appropriately conducted hyperparameter optimization using the validation set and report performance for each model based on the test set.
The authors concluded that `zimmermann-augmented-confidence` outperforms all other models across all tested metrics.
This was based on a single randomly selected testing set comprising roughly 20% of the available data.
`zimmermann` and `zimmermann-augmented` lead significantly in AUROC and are competitive with other models on other metrics.
This improvement is said to justify the increased complexity of the FART architecture.

# Methods

## Dataset Curation

On close examination of FartDB, we identified several concerns:

* Acids were assumed to be sour based only on $\mathrm{p}K_{\mathrm{a}}$. Although acidity is directly tied to sourness, there are other structural aspects that affect a compound’s perceived flavor. For instance, saccharin is a strong acid, with a $\mathrm{p}K_{\mathrm{a}}$ of 2.3, but it is extremely sweet. For this reason, it is inappropriate to assume that all acids in the pKa database taste sour. Numerous other similar examples, including caffeic acid (bitter) and L-glutamic acid (umami) are shown in Figure S1, including several compounds in FartDB[@yamamoto2023flavor; @tanase2022taste; @mattes2009there]. 
* Train-test data leakage exists for duplicate species (which were not properly identified due to differing tautomerization, salt forms, or stereochemistry) with practically all having the same flavor labels (Figure S2).
* Some labels are contradictory, wherein original data was misclassified (e.g. "not bitter" and "bitter (contradictory evidence)" labeled as "bitter"), or compounds being treated as distinct data entries if including multiple labels.

We created a __cleaned dataset__ that removes compounds derived from the $\mathrm{p}K_{\mathrm{a}}$ data, eliminates duplicates (keeping only consistent entries), discards contradictory labels, and excludes multi-label compounds. The standardization code we used is publicly available on GitHub. All standardizations were conducted using RDKit version 2025.03.4, checking for duplicate compounds based on InChI string matches after sanitizing stereochemistry, tautomerization, and salt forms.

There were other potential issues which we did not correct, but may be of interest for further analysis:

* Flavor labels were mapped to pre-defined keyword matches, which fails to capture synonyms of flavors (such as for alkyl esters shown in Figure S3). 
* Distinctions between gustatory versus olfactory data are not present in FartDB, despite both being present due to the curation from FlavorDB.

Although we have made our best efforts to clean the data as processed by Zimmermann et al., we note that there are still fundamental limitations to this data labeling approach. The somewhat arbitrary selection of categories oversimplifies the vast assortment of flavors an individual compound may have, and omits their intensities. We encourage more work to be done in general with developing comprehensive, consistent compound flavor datasets that address these issues.

## Rigorous Comparisons

Using both the original and the __cleaned dataset__, we assessed performance of the models used in Zimmermann _et al._ by retraining them from scratch.
To facilitate direct comparison, their generated models' settings are re-used where possible:

 - The XGB-based and BalancedForest models' hyperparameters (e.g., number of estimators in XGB) are copied as given in the code for the reference study.
 - `zimmermann` and `zimmermann-augmented` are trained with and without augmentation, respectively, using the same procedure as stated by the authors.
 - Chemprop is trained using its default settings in version 2.2.0; the authors stated that they used Chemprop's hyperparameter optimization procedure without modification, but do not provide the final model.

Note that _no further hyperparameter optimization is performed_, since the present analysis is intended to be on the models as presented in the reference study.

As previously discussed, the `zimmermann-augmented` model yields approximately 10 predictions for a given molecule.
To use these predictions and calculate performance metrics one must collapse the results into a single prediction (voting, weighting, etc.); this was not done in the reference study.
Instead, the authors reported performance on the entire _augmented_ testing set, an invalid comparison with other models' testing set.
To remediate this, we _train_ the `zimmermann-augmented` model with augmentation but _test_ without it, rather than propose a new scheme to collapse the multiple predictions.
While this may sacrifice the increased robustness of averaging predictions from multiple SMILES embeddings, there is simply no stated augmentation collapse strategy.

The choice of a single random split of the available data into training and testing is not sufficiently rigorous to definitively compare the presented models.
Statistical methods for comparing classifiers are well-developed [@stats98; @stats06]; we refer readers to the recent review by Rainio and coauthors [@Rainio2024] for a comprehensive tutorial on statistical comparisons in machine learning.

In the context of small-molecule property prediction, recent strides in standardizing statistical testing have been made.
To rigorously identify practical differences between the tested models, we implement the procedure laid out by Ash _et al._ [@ash]: 5x5 repeated cross validation.
This process yields 25 independent test set results for each model, with the subsequent Tukey Honestly Significant Difference test enabling simultaneous comparisons of statistically significant performance improvements across all models.

# Results

## Reanalysis of Original Dataset

Each of the models discussed above (`zimmermann-augmented`, `zimmermann`, `xgb-fingerprint-mordred`, `xgb-fingerprint`, `chemprop`, and `balancedforest`) was implemented, trained, and tested using the given procedures.

Figure 1 shows the results of the Tukey HSD test when using the _original_ FartDB as curated by Zimmermann _et al._
Each of the metrics from the reference study is included.
The statistically best performer is shown in blue with its corresponding Tukey's Q critical value as the horizontal error bar.
Models which are statistically indistinguishable at a significance level of 5% ($\alpha=0.05$) are shown in grey, with worse models shown in red.

![Figure 1. Tukey Honestly Significant Difference (HSD) test diagram comparing the multiclass classification performance by various higher-is-better metrics, given in the subtitle of each plot, based on 5x5 nested cross validation. Tested models are those presented in the reference study of Zimmermann _et al._ [@zimmermann] Models shown in blue are the best performers, with those shown in the grey being statistically indistinguishable from the  best performer according to the aforementioned test. Models shown in red are worse than the blue models. \label{tukey_hsd_original}](./figures/tukey_hsd_original.png){ width=3in }

These results introduce more nuance into the conclusions of the original manuscript regarding the FART model architecture.
`zimmermann-augmented` _does_ achieve statistically superior performance on each of the tested metrics, but it is _not_ a definitive 'winner'.
`chemprop` and `xgb-fingerprint`, in particular, are statistically indistinguishable on three of the five metrics.
Also noteworthy is that the `zimmermann` model (i.e., no augmentation) is statistically _worse_ across all tested metrics.
We further stress that these results include data leakage between the training and testing sets due to the construction of the dataset - it stands to reason that the largest model in terms of number of parameters (the FART models) could memorize the training data the best.

## Model Comparison using __Cleaned Dataset__

The above procedure was repeated on the __cleaned dataset__, yielding the results shown in Figure 2 following the same conventions of Figure 1.

![Figure 2. Tukey Honestly Significant Difference (HSD) test diagram comparing the multiclass classification performance by various higher-is-better metrics, given in the subtitle of each plot, based on 5x5 nested cross validation. Tested models are those presented in the reference study of Zimmermann _et al._ [@zimmermann] with additional data curation applied to correct errors. Models shown in blue are the best performers, with those shown in the grey being statistically indistinguishable from the  best performer according to the aforementioned test. Models shown in red are worse than the blue models. \label{tukey_hsd_refined}](./figures/tukey_hsd_sanitized.png){ width=3in }

These results do not support the conclusions of the Zimmermann _et al._ study.
`xgb-fingerprint-mordred` achieves the same performance as `zimmermann-augmented`, with both models ranking as the statistical best performer on four of the five metrics.
`zimmermann` achieves only one best performance, on par with `balancedforest`.

In summary, we thank Zimmermann _et al._ for their contributions to data availability and model development.
However, our analysis reveals that simple baselines can achieve comparable or superior performance when proper dataset curation and rigorous statistical testing are applied.
This highlights the importance of these methodological considerations in machine learning studies.

# Code Availability

All code supporting the analysis in this comment is available on [GitHub](https://github.com/jacksonburns/zimmermann_et_al_matters_arising).

# Declarations

## Acknowledgements

The authors acknowledge Prof. Sofja Tshepelevitsh and Prof. Ivo Leito for thought-provoking discussions. 
The authors further acknowledge Prof. Markus Kraft for his insightful comments in preparation of this manuscript.

## Author Contributions

J.W.B. implemented, trained, and tested models; J.W.Z. curated the revised dataset; J.W.B., J.W.Z, and W.H.G. prepared the manuscript; WHG supervised the project.

## Competing Interests

The authors declare the following competing interests: J.W.Z. is an author of a component dataset used in FartDB; J.W.B., J.W.Z., and W.H.G. are authors of the Chemprop machine learning package referenced in this comment and the original study as a baseline.

## Funding

Not applicable.

# References

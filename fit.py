import os
import warnings
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from mordred import Calculator, descriptors
from rdkit import Chem
from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.fingerprints import MorganFingerprintTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from tqdm import tqdm
from xgboost import XGBClassifier

from chemprop_estimator import ChempropMoleculeTransformer, ChempropRegressor
from zimmermann_estimator import ZimmermannClassifier

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_SEED = 101

if __name__ == "__main__":
    if os.getenv("RUN_ZIMMERMANN", False):
        print("running Zimmermann et al. model")
        run_zimmermann = True
    else:
        print("running all models except Zimmermann et al.")
        run_zimmermann = False

    # source_data = Path("data/zimmermann_et_al_flavors_original.csv")
    source_data = Path("data/zimmermann_et_al_flavors_sanitized.csv")
    outdir = Path("outputs_" + source_data.stem)
    outdir.mkdir(exist_ok=True)
    df = pd.read_csv(source_data)
    df = df.sample(frac=1.0, random_state=RANDOM_SEED)  # shuffle

    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_SEED)
    splits = tuple(rkf.split(np.arange(df.shape[0])))

    class MordredDescriptorTransformer(BaseEstimator, TransformerMixin):
        def __init__(self):
            calc = Calculator()

            calc.register(descriptors.Autocorrelation.ATSC(0, "c"))
            calc.register(descriptors.Autocorrelation.ATSC(0, "se"))
            calc.register(descriptors.Autocorrelation.AATS(0, "i"))
            calc.register(descriptors.Autocorrelation.ATSC(1, "p"))
            calc.register(descriptors.Autocorrelation.AATSC(2, "se"))
            calc.register(descriptors.Autocorrelation.AATSC(0, "m"))
            calc.register(descriptors.Autocorrelation.AATSC(1, "Z"))
            calc.register(descriptors.Autocorrelation.AATSC(2, "are"))
            calc.register(descriptors.Autocorrelation.AATSC(1, "pe"))
            calc.register(descriptors.AdjacencyMatrix.AdjacencyMatrix("SpDiam"))
            calc.register(descriptors.Autocorrelation.ATSC(1, "c"))
            calc.register(descriptors.Autocorrelation.ATSC(1, "se"))
            calc.register(descriptors.Autocorrelation.ATSC(1, "Z"))
            calc.register(descriptors.Autocorrelation.ATSC(1, "m"))
            calc.register(descriptors.Autocorrelation.ATSC(4, "s"))
            self.calc = calc

        def fit(self, X, y=None):
            return self

        def transform(self, X: np.ndarray[Chem.Mol]):
            return self.calc.pandas(X.flatten(), nmols=X.shape[0]).fill_missing(value=0.0).to_numpy(
                dtype=float
            )

    def get_pipeline(model_type: str) -> Pipeline:
        if model_type == "chemprop":
            return Pipeline(
                [
                    (str(ChempropMoleculeTransformer), ChempropMoleculeTransformer()),
                    (
                        str(ChempropRegressor),
                        ChempropRegressor(
                            multiclass_num_classes=5,
                            task_type="multiclass",
                            accelerator="cuda",
                            devices=1,
                        ),
                    ),
                ]
            )
        elif model_type == "zimmermann":
            return Pipeline(
                [(str(ZimmermannClassifier), ZimmermannClassifier(num_train_epochs=20))]
            )
        elif model_type == "zimmermann-augmented":
            return Pipeline(
                [
                    (
                        str(ZimmermannClassifier),
                        ZimmermannClassifier(
                            num_train_epochs=2,
                            perform_augmentation=True,
                            default_augmentation_number=10,
                        ),
                    )
                ]
            )
        # model
        if model_type == "xgb-fingerprint":
            model = XGBClassifier(
                n_estimators=100,
                max_depth=15,
                learning_rate=0.1,
                subsample=0.6,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=RANDOM_SEED,
                tree_method="hist",
                num_class=5,
            )
        elif model_type == "xgb-fingerprint-mordred":
            model = XGBClassifier(
                n_estimators=200,
                max_depth=15,
                learning_rate=0.1,
                subsample=0.8,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=RANDOM_SEED,
                tree_method="hist",
                num_class=5,
            )
        elif model_type == "balancedforest":
            model = BalancedRandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                criterion="gini",
                random_state=RANDOM_SEED,
                sampling_strategy="all",
                replacement=True,
                bootstrap=False,
            )
        # features
        featurizer = MorganFingerprintTransformer(fpSize=1024, radius=2)
        if model_type == "xgb-fingerprint-mordred":
            featurizer = FeatureUnion(
                [
                    (str(featurizer), featurizer),
                    ("mordred", MordredDescriptorTransformer()),
                ]
            )
        pipeline = Pipeline(
            [
                (str(SmilesToMolTransformer), SmilesToMolTransformer()),
                ("featurizer", featurizer),
                (str(model), model),
            ]
        )
        return pipeline

    for model in (
        "xgb-fingerprint",
        "zimmermann-augmented",
        "zimmermann",
        "chemprop",
        "xgb-fingerprint-mordred",
        "balancedforest",
    ):
        results = []
        if Path(outdir / f"{model}_results.csv").exists():
            print(f"Skipping {model} as results already exist.")
            continue
        if run_zimmermann and "zimmermann" not in model:
            print(f"Skipping {model} as this script was launched with accelerate")
            continue
        if not run_zimmermann and "zimmermann" in model:
            print(f"Skipping {model} as this script was launched without accelerate")
            continue
        for i, (train_idxs, test_idxs) in tqdm(
            enumerate(splits), total=len(splits), desc=f"Running {model}"
        ):
            train_df = df.iloc[train_idxs]
            test_df = df.iloc[test_idxs]
            pipeline = get_pipeline(model)
            pipeline = pipeline.fit(
                train_df["smiles"].to_numpy(), train_df["target"].to_numpy()
            )

            pred = pipeline.predict(test_df["smiles"])
            if model == "chemprop":
                pred_proba = pred.reshape(test_df.shape[0], 5)
                pred = pred_proba.argmax(1)
            else:
                pred_proba = pipeline.predict_proba(test_df["smiles"].to_numpy())

            with open(outdir / f"{model}_preds_fold_{i}.pkl", "wb") as file:
                pickle.dump(
                    {
                        "smiles": test_df["smiles"].to_numpy().flatten(),
                        "pred": pred.flatten(),
                        "pred_proba": pred_proba,
                        "target": test_df["target"].to_numpy().flatten(),
                    },
                    file,
                )

            accuracy = accuracy_score(test_df["target"], pred)
            precision = precision_score(
                test_df["target"], pred, average="macro", zero_division=0
            )
            recall = recall_score(
                test_df["target"], pred, average="macro", zero_division=0
            )
            f1 = f1_score(test_df["target"], pred, average="macro", zero_division=0)
            auroc = roc_auc_score(
                test_df["target"], pred_proba, multi_class="ovr", average="macro"
            )
            results.append(
                dict(
                    model=model,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1,
                    auroc=auroc,
                )
            )
        results_df = pd.DataFrame.from_records(results)
        results_df.to_csv(outdir / f"{model}_results.csv")

    if run_zimmermann:
        exit(
            0
        )  # Exit early if running with accelerate, as we don't want to run the plotting code
    all_results = []
    for model in (
        "zimmermann-augmented",
        "zimmermann",
        "chemprop",
        "xgb-fingerprint-mordred",
        "xgb-fingerprint",
        "balancedforest",
    ):
        all_results.append(pd.read_csv(outdir / f"{model}_results.csv"))

    results_df = pd.concat(all_results, ignore_index=True)

    # List of metrics to analyze
    metrics = ["accuracy", "precision", "recall", "f1_score", "auroc"]

    # Setup grid layout
    fig, axes = plt.subplots(nrows=len(metrics), ncols=1, figsize=(3, 6))

    # Loop through each metric
    results = {}
    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Get best model for this metric
        best_model = results_df.groupby("model")[metric].mean().idxmax()

        # Run Tukey HSD
        tukey = pairwise_tukeyhsd(
            endog=results_df[metric], groups=results_df["model"], alpha=0.05
        )

        # Plot with best model as comparison
        tukey.plot_simultaneous(comparison_name=best_model, ax=ax, figsize=(6, 12))
        ax.set_title(f"Tukey HSD for {metric}")
        ax.set_xlabel("Mean Difference")
        ax.set_ylabel("Model")
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

        _result = {}
        for model in results_df["model"].unique():
            _result[model] = f"{results_df[results_df["model"] == model][metric].mean():.3f}"
        results[metric + f" (q={tukey.halfwidths[0]:.2e})"] = _result
    print(pd.DataFrame(results).to_markdown())
    # Remove empty subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(outdir / "tukey_hsd.png")

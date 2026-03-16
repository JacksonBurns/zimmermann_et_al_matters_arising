import itertools
import logging
import math
from collections import Counter

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Set up logging to capture RDKit warnings and other messages
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Import RDKit Chem only if needed. It's used in _smiles_to_random.
try:
    from rdkit import Chem
except ImportError:
    logger.error(
        "RDKit not found. Please install it to enable SMILES augmentation (pip install rdkit-pypi)."
    )
    Chem = None  # Set to None if RDKit is not available


class ZimmermannClassifier(BaseEstimator, ClassifierMixin):
    """
    A Scikit-learn compatible estimator for classifying SMILES strings using a
    Hugging Face Transformers model. This version assumes target labels (y) are
    already integer-encoded (0 to num_labels - 1).

    The fit method now includes a random split for a validation set and
    optionally performs SMILES augmentation on the training data.
    """

    def __init__(
        self,
        model_checkpoint="seyonec/SMILES_tokenized_PubChem_shard00_160k",
        num_labels=5,  # IMPORTANT: This must match the actual number of unique classes (0-4 means 5 classes)
        max_length=512,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        evaluation_strategy="steps",  # Keep as steps or epoch for monitoring
        save_strategy="steps",
        load_best_model_at_end=True,  # Essential for reverting to best state
        save_total_limit=5,
        output_dir="./results",
        logging_dir="./logs",
        run_name="SMILES_Classifier_Run",
        validation_split_percentage=0.176,  # Parameter for validation split
        metric_for_best_model="accuracy",  # Metric to monitor for early stopping
        greater_is_better=True,  # For accuracy, higher is better
        # Augmentation parameters
        perform_augmentation=False,  # New: Whether to perform augmentation
        augmentation_map=None,  # New: Augmentation rules per class
        default_augmentation_number=0,  # New: Default augmentation for unspecified classes
    ):
        """
        Initializes the SMILESClassifier.

        Args:
            model_checkpoint (str): Pretrained model checkpoint to use.
            num_labels (int): The exact number of unique integer labels (e.g., 5 for labels 0-4).
                              It is crucial that this matches the actual range of your input `y`.
            max_length (int): Maximum sequence length for tokenization.
            num_train_epochs (int): Number of training epochs.
            per_device_train_batch_size (int): Batch size per device for training.
            per_device_eval_batch_size (int): Batch size per device for evaluation.
            weight_decay (float): Weight decay for optimizer.
            evaluation_strategy (str): Strategy for evaluation during training (e.g., "steps", "epoch", "no").
            save_strategy (str): Strategy for saving checkpoints during training (e.g., "steps", "epoch", "no").
            load_best_model_at_end (bool): Whether to load the best model at the end of training.
            save_total_limit (int): Maximum number of checkpoints to keep.
            output_dir (str): Output directory for model checkpoints and predictions.
            logging_dir (str): Directory for logging.
            run_name (str): Name for the Weights & Biases run (if integrated).
            validation_split_percentage (float): The percentage of the input data to use as a validation set.
                                                 Must be between 0.0 and 1.0.
            metric_for_best_model (str): The metric to monitor for determining the best model.
            greater_is_better (bool): Whether a greater value of `metric_for_best_model` is better.
            perform_augmentation (bool): If True, SMILES augmentation will be applied to the training data.
            augmentation_map (dict, optional): A dictionary where keys are integer
                labels (e.g., 0, 1, 2, 3, 4) and values are the number of unique
                random SMILES to generate for each original SMILES belonging to
                that class. If a class is not in the map, it will use
                default_augmentation_number.
                Example: {0: 10, 1: 5, 2: 0}
            default_augmentation_number (int): The number of unique random SMILES
                to generate for classes not specified in `augmentation_map`.
                If 0, no augmentation will occur for unspecified classes.
        """
        self.model_checkpoint = model_checkpoint
        self.num_labels = num_labels
        self.max_length = max_length
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.weight_decay = weight_decay
        self.evaluation_strategy = evaluation_strategy
        self.save_strategy = save_strategy
        self.load_best_model_at_end = load_best_model_at_end
        self.save_total_limit = save_total_limit
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        self.run_name = run_name
        self.validation_split_percentage = validation_split_percentage
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        self.perform_augmentation = perform_augmentation
        self.augmentation_map = augmentation_map if augmentation_map is not None else {}
        self.default_augmentation_number = default_augmentation_number

        # Initialize tokenizer here, but model and trainer in fit
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = None
        self.trainer = None

    @staticmethod
    def _control_smiles_duplication(random_smiles, duplicate_control=lambda x: 1):
        """
        Returns augmented SMILES with the number of duplicates controlled by the function duplicate_control.
        Used internally to ensure unique SMILES when duplicate_control is lambda x: 1.

        Parameters
        ----------
        random_smiles : list
            A list of random SMILES.
        duplicate_control : func, Optional, default: lambda x: 1
            The number of times a SMILES will be duplicated, as function of the number of times
            it was included in `random_smiles`. This number is rounded up to the nearest integer.

        Returns
        -------
        list
            A list of random SMILES with duplicates controlled.
        """
        if not random_smiles:
            return []

        counted_smiles = Counter(random_smiles)
        smiles_duplication = {
            smiles: math.ceil(duplicate_control(counted_smiles[smiles]))
            for smiles in counted_smiles
        }
        return list(
            itertools.chain.from_iterable(
                [[smiles] * smiles_duplication[smiles] for smiles in smiles_duplication]
            )
        )

    def _smiles_to_random(self, smiles, num_augmentations=50):
        """
        Takes a SMILES (not necessarily canonical) and returns `num_augmentations`
        random variations of this SMILES.

        Parameters
        ----------
        smiles : str
            SMILES string describing a compound.
        num_augmentations : int, Optional, default: 50
            The number of random SMILES generated.

        Returns
        -------
        list
            A list of `num_augmentations` random (may not be unique) SMILES
            or None if the initial SMILES is not valid.
        """
        if Chem is None:
            logger.error("RDKit is not installed. Cannot perform SMILES augmentation.")
            return None

        if not isinstance(smiles, str):
            logger.warning(
                f"Invalid SMILES input (not a string): {smiles}. Skipping augmentation."
            )
            return None

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(
                f"Could not parse SMILES: '{smiles}'. Skipping augmentation."
            )
            return None
        else:
            if num_augmentations > 0:
                return [
                    Chem.MolToSmiles(mol, canonical=False, doRandom=True)
                    for _ in range(num_augmentations)
                ]
            elif num_augmentations == 0:
                return [smiles]  # Return original if no augmentation requested
            else:
                raise ValueError("num_augmentations must be greater or equal to zero.")

    def _augment_single_smiles(self, smiles, num_augmentations):
        """
        Generates a list of unique random SMILES for a single input SMILES.

        Parameters
        ----------
        smiles : str
            SMILES string describing a compound.
        num_augmentations : int
            The number of random SMILES to attempt to generate.

        Returns
        -------
        list
            A list of unique random SMILES (no duplicates).
        """
        random_smiles_list = self._smiles_to_random(smiles, num_augmentations)
        if random_smiles_list is None:
            return []  # Return empty list if SMILES is invalid or RDKit not available
        return self._control_smiles_duplication(random_smiles_list, lambda x: 1)

    def _perform_augmentation(self, X, y):
        """
        Transforms the input SMILES and labels by augmenting them based on
        the `augmentation_map`.

        Args:
            X (list or np.array): A list of SMILES strings.
            y (list or np.array): Corresponding integer-encoded labels (0 to num_labels - 1).

        Returns:
            tuple: A tuple containing two numpy arrays:
                   - augmented_X (np.array): Augmented list of SMILES strings.
                   - augmented_y (np.array): Augmented list of corresponding integer labels.
        """
        if not isinstance(X, (list, np.ndarray)) or not isinstance(
            y, (list, np.ndarray)
        ):
            raise TypeError("X and y must be lists or numpy arrays.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")

        augmented_smiles = []
        augmented_labels = []

        # Convert to list for easier iteration if they are numpy arrays
        X_list = list(X)
        y_list = list(y)

        for i, (smiles, label) in enumerate(zip(X_list, y_list)):
            augmentation_number = self.augmentation_map.get(
                label, self.default_augmentation_number
            )

            if augmentation_number > 0:
                new_smiles_list = self._augment_single_smiles(
                    smiles, augmentation_number
                )
                # Add the original SMILES and its label
                augmented_smiles.append(smiles)
                augmented_labels.append(label)
                # Add the augmented SMILES and their labels
                for new_s in new_smiles_list:
                    augmented_smiles.append(new_s)
                    augmented_labels.append(label)
            else:
                # If no augmentation for this class, just keep the original entry
                augmented_smiles.append(smiles)
                augmented_labels.append(label)

        return np.array(augmented_smiles), np.array(augmented_labels)

    def _tokenize_function(self, examples):
        """
        Tokenizes SMILES strings.
        """
        return self.tokenizer(
            examples["smiles"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

    def _compute_metrics(self, eval_pred):
        """
        Computes accuracy metrics for evaluation predictions.
        """
        metric = evaluate.load("accuracy")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    def fit(self, X, y):
        """
        Fits the SMILES classification model.

        Args:
            X (list or np.array): A list of SMILES strings.
            y (list or np.array): Pre-encoded integer target labels (0 to num_labels - 1).

        Returns:
            self: The fitted estimator.
        """
        # y is already assumed to be integer-encoded
        encoded_labels = y

        # --- Augmentation Step (if enabled) ---
        if self.perform_augmentation:
            logger.info("Performing SMILES augmentation on training data...")
            X_augmented, y_augmented = self._perform_augmentation(X, encoded_labels)
            logger.info(
                f"Original training data size: {len(X)}. Augmented training data size: {len(X_augmented)}"
            )
            X_for_training = X_augmented
            y_for_training = y_augmented
        else:
            X_for_training = X
            y_for_training = encoded_labels
        # -------------------------------------

        # Create a pandas DataFrame for easier conversion to Hugging Face Dataset
        df = pd.DataFrame({"smiles": X_for_training, "label": y_for_training})

        # Split the data into training and validation sets
        # Use stratify=encoded_labels to maintain class distribution if possible
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            df["smiles"],
            df["label"],
            test_size=self.validation_split_percentage,
            random_state=42,  # For reproducibility
            stratify=(
                df["label"] if len(np.unique(df["label"])) > 1 else None
            ),  # Stratify only if multiple classes
        )

        train_df = pd.DataFrame({"smiles": X_train_split, "label": y_train_split})
        val_df = pd.DataFrame({"smiles": X_val_split, "label": y_val_split})

        train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
        eval_dataset = Dataset.from_pandas(val_df, preserve_index=False)

        # Tokenize the datasets
        train_dataset = train_dataset.map(self._tokenize_function, batched=True)
        eval_dataset = eval_dataset.map(self._tokenize_function, batched=True)

        # Remove unnecessary columns for training/evaluation
        train_dataset = train_dataset.remove_columns(["smiles"])
        eval_dataset = eval_dataset.remove_columns(["smiles"])

        # Initialize the model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_checkpoint, num_labels=self.num_labels
        )

        # Define training arguments
        training_args = TrainingArguments(
            run_name=self.run_name,
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            weight_decay=self.weight_decay,
            eval_strategy=self.evaluation_strategy,
            logging_dir=self.logging_dir,
            save_strategy=self.save_strategy,
            load_best_model_at_end=self.load_best_model_at_end,
            save_total_limit=self.save_total_limit,
            metric_for_best_model=self.metric_for_best_model,  # Explicitly set metric for best model
            greater_is_better=self.greater_is_better,  # Explicitly set if greater is better
        )

        # Initialize the Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,  # Pass the evaluation dataset here
            compute_metrics=self._compute_metrics,
        )

        # Train the model
        self.trainer.train()

        return self

    def predict(self, X):
        """
        Predicts labels for new SMILES strings.

        Args:
            X (list or np.array): A list of SMILES strings to predict.

        Returns:
            np.array: Predicted integer labels (0 to num_labels - 1).
        """
        if self.trainer is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Create a pandas DataFrame for prediction
        df_predict = pd.DataFrame({"smiles": X})
        predict_dataset = Dataset.from_pandas(df_predict, preserve_index=False)

        # Tokenize the prediction dataset
        predict_dataset = predict_dataset.map(self._tokenize_function, batched=True)

        # Remove unnecessary columns for prediction
        predict_dataset = predict_dataset.remove_columns(["smiles"])

        # Make predictions
        predictions = self.trainer.predict(predict_dataset)
        pred_logits = predictions.predictions
        pred_labels_encoded = np.argmax(pred_logits, axis=-1)

        # Labels are already in the desired integer format
        return pred_labels_encoded

    def predict_proba(self, X):
        """
        Predicts class probabilities for new SMILES strings.

        Args:
            X (list or np.array): A list of SMILES strings to predict probabilities for.

        Returns:
            np.array: Predicted class probabilities.
        """
        if self.trainer is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Create a pandas DataFrame for probability prediction
        df_predict = pd.DataFrame({"smiles": X})
        predict_dataset = Dataset.from_pandas(df_predict, preserve_index=False)

        # Tokenize the prediction dataset
        predict_dataset = predict_dataset.map(self._tokenize_function, batched=True)

        # Remove unnecessary columns for prediction
        predict_dataset = predict_dataset.remove_columns(["smiles"])

        # Make predictions
        predictions = self.trainer.predict(predict_dataset)
        pred_logits = predictions.predictions

        # Apply softmax to get probabilities
        probabilities = softmax(pred_logits, axis=1)

        return probabilities

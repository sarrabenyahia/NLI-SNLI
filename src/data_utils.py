import pandas as pd
from transformers import AutoTokenizer, AutoConfig
from sklearn.base import TransformerMixin, BaseEstimator
import torch
from sklearn.preprocessing import LabelEncoder


class Filter(TransformerMixin, BaseEstimator):
    def __init__(self) -> None:
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_filtered = X.filter(lambda x: x["label"] != -1)

        return X_filtered


class PreProcessor(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        model_name="cross-encoder/nli-roberta-base",
        max_length=128,
        truncation=True,
        padding=True,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding
        self.label_encoder = LabelEncoder()

    def fit(self, X, y=None):
        self.label_encoder.fit(X["label"])
        return self

    def transform(self, X):
        labels = self.label_encoder.transform(X["label"])
        encodings = self.tokenizer(
            X["premise"],
            X["hypothesis"],
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
        )

        # Convert inputs and labels to list of dictionaries
        dataset_inputs = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }
        dataset_data = []
        for i in range(len(encodings["input_ids"])):
            dataset_data.append(
                {key: torch.tensor(val[i]) for key, val in dataset_inputs.items()}
            )
        dataset_data = [
            {
                "input_ids": input["input_ids"],
                "attention_mask": input["attention_mask"],
                "labels": label,
            }
            for input, label in zip(dataset_data, labels)
        ]

        return dataset_data, labels

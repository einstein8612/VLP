from typing import Self

import torch
from torch.utils.data import Dataset

class BaseModel:
    def fit(self, dataset: Dataset):
        """
        Fit the model to the dataset.

        :param dataset: The dataset to fit the model to.
        """
        raise NotImplementedError("Fit method not implemented.")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict using the model on data.

        :param X: The data to predict on.
        :return: The predictions.
        """
        raise NotImplementedError("Predict method not implemented.")

    def save(self, model_path: str) -> str:
        """
        Save the model to the specified path.

        :param model_path: The path to save the model.
        :return: The path where the model is saved.
        """
        raise NotImplementedError("Save method not implemented.")

    def load(self, model_path: str) -> Self:
        """
        Load the model from the specified path.

        :param model_path: The path to load the model from.
        :return: The loaded model.
        """
        raise NotImplementedError("Load method not implemented.")
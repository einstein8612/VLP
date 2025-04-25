import pickle

import torch
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import DataLoader

from dataset import FPDataset
from models.base import BaseModel


class RF(BaseModel):
    def __init__(self, n_estimators=100, max_depth=None, seed=None):
        """
        Initialize the Random Forest model.

        :param n_estimators: The number of trees in the forest.
        :param max_depth: The maximum depth of the tree.
        :param seed: Controls the randomness of the estimator.
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, verbose=2, n_jobs=-1, random_state=seed
        )

    def fit(self, dataset: FPDataset):
        """
        Fit the model to the dataset.

        :param dataset: The dataset to fit the model to.
        """

        loader = DataLoader(dataset, batch_size=len(dataset))
        X_tensor, y_tensor = next(iter(loader))
        X = X_tensor.numpy()
        y = y_tensor.numpy()

        self.model.fit(X, y)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict using the model on the dataset.

        :param dataset: The dataset to predict on.
        :return: The predictions.
        """
        return torch.from_numpy(self.model.predict(X.numpy()))

    def save(self, model_path: str):
        """
        Save the model to the specified path using pickle.

        :param model_path: The path to save the model.
        """
        with open(model_path+".pickle", "wb") as f:
            pickle.dump(self.model, f)
    
    def load(model_path: str) -> "RF":
        """
        Load the model from the specified path using pickle.

        :param model_path: The path to load the model from.
        :return: The loaded model.
        """
        with open(model_path, "rb") as f:
            rf = RF()
            rf.model = pickle.load(f)
            return rf
        raise ValueError(f"Model at {model_path} not loaded")

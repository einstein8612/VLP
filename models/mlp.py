import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FPDataset
from models.base import BaseModel


class MLP(BaseModel):
    def __init__(self, batch_size=256, lr=0.001, epochs=100, seed=None):
        """
        Initialize the MLP model.

        :param seed: Controls the randomness of the estimator.
        """
        self.model = nn.Sequential(
            nn.Linear(36, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.seed = seed

    def fit(self, dataset: FPDataset):
        """
        Fit the model to the dataset.

        :param dataset: The dataset to fit the model to.
        """
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        torch.manual_seed(self.seed)

        # Training loop
        self.model.train()
        for _ in tqdm(range(self.epochs), desc="Training MLP", unit="epoch"):
            for X, y in loader:
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs, y.float())
                loss.backward()
                optimizer.step()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict using the model on the dataset.

        :param dataset: The dataset to predict on.
        :return: The predictions.
        """

        return self.model.forward(X)

    def save(self, model_path: str):
        """
        Save the model to the specified path using pickle.

        :param model_path: The path to save the model.
        """
        model_path = model_path + ".pth"
        torch.save(self.model.state_dict(), model_path)
        return model_path

    def load(model_path: str) -> "MLP":
        """
        Load the model from the specified path using pickle.

        :param model_path: The path to load the model from.
        :return: The loaded model.
        """
        with open(model_path, "rb") as f:
            mlp = MLP()
            mlp.model.load_state_dict(torch.load(f))
            return mlp
        raise ValueError(f"Model at {model_path} not loaded")

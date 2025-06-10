import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FPDataset
from models.base import BaseModel


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        

class NormalizeInput(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=1, keepdim=True) + 1e-8)

class MLP(BaseModel):
    def __init__(self, batch_size=64, lr=0.001, epochs=25, normalize=False, device="cpu", seed=None):
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
        ).to(device)

        # Normalize input if necessary        
        if normalize:
            self.model.insert(0, NormalizeInput())

        self.model.apply(init_weights)

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
                # loss = (torch.norm(outputs - y.float(), dim=1)**2).mean()
                loss = criterion(outputs, y.float())
                loss.backward()
                optimizer.step()

    def predict(self, X: torch.Tensor, eval: bool=False) -> torch.Tensor:
        """
        Predict using the model on the dataset.

        :param X: The data to predict on.
        :param eval: Whether or not it's in evaluation mode.
        :return: The predictions.
        """
        self.model.eval()
        return self.model.forward(X)

    def save(self, model_path: str):
        """
        Save the model to the specified path using pickle.

        :param model_path: The path to save the model.
        """
        model_path = model_path + ".pth"
        torch.save(self.model.state_dict(), model_path)
        return model_path

    def load(self, model_path: str):
        """
        Load the model from the specified path using pickle.

        :param model_path: The path to load the model from.
        :return: The loaded model.
        """
        with open(model_path, "rb") as f:
            self.model.load_state_dict(torch.load(f))
            return
        raise ValueError(f"Model at {model_path} not loaded")

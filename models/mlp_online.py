import numpy as np
import torch
from ransac_line import fit as fit_ransac_line
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

class MLPOnline(BaseModel):
    def __init__(self, data_npy_path: str, batch_size=256, lr=0.001, epochs=250, device="cpu", seed=None):
        """
        Initialize the MLP model.

        :param seed: Controls the randomness of the estimator.
        """
        self.model = nn.Sequential(
            NormalizeInput(),

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

        self.model.apply(init_weights)

        self.data = torch.tensor(np.load(data_npy_path), dtype=torch.float32, device=device)
        self.scalars = torch.ones(self.data.shape[2], dtype=torch.float32).to(device)
        self.min_positions = torch.zeros(2, dtype=torch.long, device=device)
        self.max_positions = torch.tensor([self.data.shape[1] - 1, self.data.shape[0] - 1], dtype=torch.long, device=device)

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

        # Apply the scalars to the input data
        X = X * self.scalars
        # Make predictions
        predictions = self.model.forward(X)

        # If in evaluation mode, return the predictions directly
        if eval:
            return predictions
        
        # If in online learning mode, apply RANSAC to the predictions to refine the scalars
        # before returning the predictions
        
        # Calculate the positions in the data array
        positions = torch.clamp(torch.round(predictions / 10).long(), min=self.min_positions, max=self.max_positions)

        # Extract the references from the data array using the positions
        references = self.data[positions[:, 1], positions[:, 0]]
        # Only consider references that are not -1, which indicates invalid data
        mask = torch.all(references != -1, axis=1)

        # Use RANSAC to refine the scalars
        for i in range(36):
            self.scalars[i] *= fit_ransac_line(X[mask, i].cpu().numpy().astype(np.float32), references[mask, i].cpu().numpy().astype(np.float32), threshold=1.0, max_iters=10, seed=self.seed)

        return predictions

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
